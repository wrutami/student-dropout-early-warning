"""Prototipe deteksi dini dropout mahasiswa Jaya Jaya Institut (Streamlit).

Jalankan lokal: streamlit run app.py
Model: Gradient Boosting, memakai data pendaftaran, keuangan, dan hasil semester 1.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
MODEL_PATH = BASE / "model" / "dropout_model.joblib"
META_PATH = BASE / "model" / "feature_meta.json"

YES_NO = {1: "Ya", 0: "Tidak"}
MARITAL = {1: "Single", 2: "Menikah", 3: "Duda/janda", 4: "Bercerai", 5: "Hidup bersama (facto union)",
           6: "Pisah secara hukum"}
APPLICATION_MODE = {1: "1st phase - general contingent", 2: "Ordinance No. 612/93",
                    5: "1st phase - special contingent (Azores)", 7: "Holders of other higher courses",
                    10: "Ordinance No. 854-B/99", 15: "International student (bachelor)",
                    16: "1st phase - special contingent (Madeira)", 17: "2nd phase - general contingent",
                    18: "3rd phase - general contingent", 26: "Ordinance No. 533-A/99 b2 (Different Plan)",
                    27: "Ordinance No. 533-A/99 b3 (Other Institution)", 39: "Over 23 years old",
                    42: "Transfer", 43: "Change of course", 44: "Technological specialization diploma holders",
                    51: "Change of institution/course", 53: "Short cycle diploma holders",
                    57: "Change of institution/course (International)"}
PREV_QUALIFICATION = {1: "Secondary education", 2: "Higher education - bachelor's degree",
                      3: "Higher education - degree", 4: "Higher education - master's",
                      5: "Higher education - doctorate", 6: "Frequency of higher education",
                      9: "12th year - not completed", 10: "11th year - not completed",
                      12: "Other - 11th year", 14: "10th year", 15: "10th year - not completed",
                      19: "Basic education 3rd cycle", 38: "Basic education 2nd cycle",
                      39: "Technological specialization course", 40: "Higher education - degree (1st cycle)",
                      42: "Professional higher technical course", 43: "Higher education - master (2nd cycle)"}


@st.cache_resource
def load_artifacts():
    return joblib.load(MODEL_PATH), json.loads(META_PATH.read_text())


def risk_level(prob):
    if prob >= 0.6:
        return "Tinggi", "🔴"
    if prob >= 0.3:
        return "Sedang", "🟠"
    return "Rendah", "🟢"


def warning_signals(row):
    """Sinyal risiko dari temuan EDA, untuk menjelaskan hasil prediksi kepada pembimbing."""
    signals = []
    if row["Tuition_fees_up_to_date"] == 0:
        signals.append("UKT tidak lancar (94% mahasiswa seperti ini dropout)")
    if row["Debtor"] == 1:
        signals.append("Memiliki utang ke kampus (76% dropout)")
    approved, enrolled = row["Curricular_units_1st_sem_approved"], row["Curricular_units_1st_sem_enrolled"]
    if approved <= 2:
        signals.append(f"Hanya lulus {approved:.0f} SKS di semester 1 (88–97% dropout)")
    elif enrolled and approved / enrolled < 0.6:
        signals.append(f"Lulus {approved:.0f} dari {enrolled:.0f} SKS yang diambil di semester 1")
    if row["Curricular_units_1st_sem_grade"] < 11:
        signals.append(f"Nilai rata-rata semester 1 rendah ({row['Curricular_units_1st_sem_grade']:.1f}/20)")
    if row["Scholarship_holder"] == 0:
        signals.append("Bukan penerima beasiswa (penerima beasiswa hanya 14% dropout)")
    if row["Age_at_enrollment"] >= 23:
        signals.append(f"Masuk kuliah di usia {row['Age_at_enrollment']:.0f} tahun (usia 23–35 dropout 59–67%)")
    if row["Daytime_evening_attendance"] == 0:
        signals.append("Kuliah kelas malam (51% dropout)")
    return signals


def choice(label, options, default, key, help_text=None):
    keys = list(options)
    index = keys.index(int(default)) if int(default) in keys else 0
    return st.selectbox(label, keys, index=index, format_func=lambda k: options[k], key=key, help=help_text)


def number(label, col, meta, step=1.0, key=None, help_text=None):
    low, high = meta["ranges"][col]
    is_int = step == 1.0 and float(meta["defaults"][col]).is_integer()
    if is_int:
        return st.number_input(label, int(low), int(high), int(meta["defaults"][col]), step=1, key=key, help=help_text)
    return st.number_input(label, float(low), float(high), float(meta["defaults"][col]), step=step, key=key,
                           help=help_text)


def student_form(meta):
    d = meta["defaults"]
    course_options = {int(k): v for k, v in meta["course"].items()}
    data = {}

    st.subheader("1. Data pendaftaran")
    c1, c2, c3 = st.columns(3)
    with c1:
        data["Course"] = choice("Program studi", course_options, d["Course"], "course")
        data["Daytime_evening_attendance"] = choice("Jadwal kuliah", {1: "Siang", 0: "Malam"},
                                                    d["Daytime_evening_attendance"], "att")
        data["Application_mode"] = choice("Jalur pendaftaran", APPLICATION_MODE, d["Application_mode"], "appmode")
    with c2:
        data["Age_at_enrollment"] = number("Usia saat masuk", "Age_at_enrollment", meta, key="age")
        data["Gender"] = choice("Jenis kelamin", {0: "Perempuan", 1: "Laki-laki"}, d["Gender"], "gender")
        data["Marital_status"] = choice("Status pernikahan", MARITAL, d["Marital_status"], "marital")
    with c3:
        data["Admission_grade"] = number("Nilai masuk (0–200)", "Admission_grade", meta, step=0.1, key="adm")
        data["Previous_qualification_grade"] = number("Nilai pendidikan sebelumnya (0–200)",
                                                      "Previous_qualification_grade", meta, step=0.1, key="prevg")
        data["Application_order"] = number("Urutan pilihan prodi (0 = pilihan pertama)", "Application_order",
                                           meta, key="order")

    st.subheader("2. Keuangan & beasiswa")
    c1, c2, c3 = st.columns(3)
    with c1:
        data["Tuition_fees_up_to_date"] = choice("UKT dibayar lancar?", YES_NO, d["Tuition_fees_up_to_date"], "ukt")
    with c2:
        data["Debtor"] = choice("Memiliki utang ke kampus?", YES_NO, d["Debtor"], "debtor")
    with c3:
        data["Scholarship_holder"] = choice("Penerima beasiswa?", YES_NO, d["Scholarship_holder"], "schol")

    st.subheader("3. Hasil akademik semester 1")
    c1, c2, c3 = st.columns(3)
    with c1:
        data["Curricular_units_1st_sem_enrolled"] = number("SKS/mata kuliah diambil",
                                                           "Curricular_units_1st_sem_enrolled", meta, key="enr")
        data["Curricular_units_1st_sem_approved"] = number("SKS/mata kuliah lulus",
                                                           "Curricular_units_1st_sem_approved", meta, key="app")
    with c2:
        data["Curricular_units_1st_sem_grade"] = number("Nilai rata-rata (0–20)", "Curricular_units_1st_sem_grade",
                                                        meta, step=0.1, key="grade")
        data["Curricular_units_1st_sem_evaluations"] = number("Jumlah evaluasi/ujian diikuti",
                                                              "Curricular_units_1st_sem_evaluations", meta, key="eval")
    with c3:
        data["Curricular_units_1st_sem_credited"] = number("SKS dikreditkan (transfer)",
                                                           "Curricular_units_1st_sem_credited", meta, key="cred")
        data["Curricular_units_1st_sem_without_evaluations"] = number(
            "Mata kuliah tanpa evaluasi", "Curricular_units_1st_sem_without_evaluations", meta, key="woeval")

    with st.expander("Data lain (opsional, terisi nilai yang paling umum)"):
        c1, c2, c3 = st.columns(3)
        with c1:
            data["Previous_qualification"] = choice("Pendidikan sebelumnya", PREV_QUALIFICATION,
                                                    d["Previous_qualification"], "prevq")
            data["Displaced"] = choice("Mahasiswa perantau (displaced)?", YES_NO, d["Displaced"], "disp")
            data["Educational_special_needs"] = choice("Berkebutuhan khusus?", YES_NO,
                                                       d["Educational_special_needs"], "sen")
            data["International"] = choice("Mahasiswa internasional?", YES_NO, d["International"], "intl")
        with c2:
            for col, label in [("Nacionality", "Kode kewarganegaraan"), ("Mothers_qualification", "Kode pendidikan ibu"),
                               ("Fathers_qualification", "Kode pendidikan ayah"),
                               ("Mothers_occupation", "Kode pekerjaan ibu"), ("Fathers_occupation", "Kode pekerjaan ayah")]:
                data[col] = number(label, col, meta, key=col, help_text="Kode sesuai dokumentasi dataset")
        with c3:
            data["Unemployment_rate"] = number("Tingkat pengangguran (%)", "Unemployment_rate", meta, step=0.1, key="unemp")
            data["Inflation_rate"] = number("Tingkat inflasi (%)", "Inflation_rate", meta, step=0.1, key="infl")
            data["GDP"] = number("GDP", "GDP", meta, step=0.01, key="gdp")
    return pd.DataFrame([data])[meta["features"]]


def show_result(model, row):
    prob = float(model.predict_proba(row)[0, 1])
    level, icon = risk_level(prob)
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Skor risiko dropout", f"{prob:.0%}")
        st.markdown(f"### {icon} Risiko {level}")
    with c2:
        signals = warning_signals(row.iloc[0])
        if signals:
            st.markdown("**Sinyal yang perlu diperhatikan:**")
            for s in signals:
                st.markdown(f"- {s}")
        else:
            st.markdown("Tidak ada sinyal risiko utama yang terdeteksi.")
    if level == "Tinggi":
        st.error("Rekomendasi: jadwalkan sesi bimbingan akademik dan cek kondisi keuangan mahasiswa secepatnya.")
    elif level == "Sedang":
        st.warning("Rekomendasi: pantau nilai dan kehadiran di semester berikutnya, tawarkan tutor sebaya.")
    else:
        st.success("Rekomendasi: cukup pemantauan rutin.")
    st.caption("Skor adalah peringkat prioritas bimbingan dari model, bukan peluang pasti.")


def batch_tab(model, meta):
    st.write("Unggah CSV dengan kolom yang sama seperti dataset (pemisah `;` atau `,`). "
             "Hasilnya diurutkan dari risiko tertinggi dan bisa diunduh.")
    file = st.file_uploader("CSV data mahasiswa", type="csv")
    if file is None:
        return
    raw = pd.read_csv(file, sep=None, engine="python")
    missing = [c for c in meta["features"] if c not in raw.columns]
    if missing:
        st.error(f"Kolom tidak ditemukan: {', '.join(missing)}")
        return
    result = raw.copy()
    result["dropout_probability"] = model.predict_proba(raw[meta["features"]])[:, 1].round(3)
    result["risk_level"] = result["dropout_probability"].apply(lambda p: risk_level(p)[0])
    result = result.sort_values("dropout_probability", ascending=False)
    counts = result["risk_level"].value_counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 Risiko tinggi", int(counts.get("Tinggi", 0)))
    c2.metric("🟠 Risiko sedang", int(counts.get("Sedang", 0)))
    c3.metric("🟢 Risiko rendah", int(counts.get("Rendah", 0)))
    st.dataframe(result, use_container_width=True, height=400)
    st.download_button("Unduh hasil prediksi (CSV)", result.to_csv(index=False).encode("utf-8"),
                       "prediksi_dropout.csv", "text/csv")


def main():
    st.set_page_config(page_title="Deteksi Dini Dropout", page_icon="🎓", layout="wide")
    model, meta = load_artifacts()
    st.title("🎓 Deteksi Dini Risiko Dropout Mahasiswa")
    st.write("Prototipe untuk **Jaya Jaya Institut**. Masukkan data pendaftaran, keuangan, dan hasil **semester 1** "
             "untuk memperkirakan risiko dropout, supaya mahasiswa berisiko bisa segera mendapat bimbingan.")
    single, batch = st.tabs(["Satu mahasiswa", "Banyak mahasiswa (CSV)"])
    with single:
        with st.form("student"):
            row = student_form(meta)
            submitted = st.form_submit_button("Prediksi risiko", type="primary")
        if submitted:
            show_result(model, row)
    with batch:
        batch_tab(model, meta)
    st.divider()
    st.caption("Model: Gradient Boosting (ROC-AUC 0,95; recall dropout 88% pada data uji). "
               "Dilatih dari data Dropout vs Graduate tanpa fitur semester 2.")


if __name__ == "__main__":
    main()
