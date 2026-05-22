# 🔍 Fraud Detection using Machine Learning

> Detecting fraudulent financial transactions from a highly imbalanced dataset using Logistic Regression with a full scikit-learn Pipeline — with a live **Streamlit Web App** for real-time predictions.

---

## 🌐 Live Demo
👉 [Click here to try the app](https://financialfraud-detection-uwnyexdan7jybczwqxnrdg.streamlit.app

## 📌 Problem Statement

Financial fraud is a critical real-world problem. This project builds a machine learning model to automatically detect fraudulent transactions from over **6.3 million records**, where only **0.13%** are actual fraud — making it a classic imbalanced classification challenge.

---

## 📊 Dataset

- **File:** `AIML Dataset.csv`
- **Source:**  https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset
- **Size:** ~6.3 million transactions
- **Target Column:** `isFraud` (0 = Legitimate, 1 = Fraud)

| Label | Count |
|-------|-------|
| Not Fraud (0) | 6,354,407 |
| Fraud (1) | 8,213 |


---

## 🗂️ Project Structure

```
fraud-detection/
│
├── analysis_model.ipynb          ← Main Jupyter Notebook (EDA + Model)
├── Fraud_DETECTION.py            ← Streamlit Web App for live predictions
├── fraud_detection_pipeline.pkl  ← Saved trained model
├── AIML Dataset.csv              ← Dataset
└── README.md                     ← Project documentation
```

---

## 🔎 Exploratory Data Analysis (EDA)

The notebook covers the following analysis:

- ✅ Class distribution of `isFraud` and `isFlaggedFraud`
- ✅ Transaction type breakdown (Bar chart)
- ✅ Fraud rate by transaction type (Bar chart)
- ✅ Transaction amount distribution (Histogram - log scale)
- ✅ Amount vs Fraud boxplot (filtered under 50k)
- ✅ Balance difference engineering for senders & receivers
- ✅ Frauds over time/step (Line chart)
- ✅ Top 10 senders and receivers
- ✅ Correlation matrix & Heatmap
- ✅ Accounts that became zero after TRANSFER/CASH_OUT

### Key Insight 💡
Almost all frauds happen in **TRANSFER** and **CASH_OUT** transaction types, and the sender's account balance often drops to **zero** after the transaction.

---

## ⚙️ Feature Engineering

Two new features were created:

```python
df["balanceDiffOrg"]  = df["oldbalanceOrg"]  - df["newbalanceOrig"]
df["balanceDiffDest"] = df["oldbalanceDest"] - df["newbalanceDest"]
```

Dropped columns: `nameOrig`, `nameDest`, `isFlaggedFraud`, `step`

---

## 🤖 Machine Learning Model

### Pipeline Architecture

```
Input Features
     ↓
ColumnTransformer
  ├── StandardScaler     → numerical columns
  └── OneHotEncoder      → categorical column (type)
     ↓
LogisticRegression
  └── class_weight="balanced"   ← handles imbalance
```

### Features Used

| Type | Columns |
|------|---------|
| Numerical | `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest` |
| Categorical | `type` |

### Train/Test Split

- **Test size:** 30%
- **Stratified:** Yes (preserves fraud ratio)

---

## 📈 Model Evaluation

Evaluated using:
- `classification_report` — Precision, Recall, F1-Score
- `confusion_matrix` — TP, TN, FP, FN
- `pipeline.score()` — Overall accuracy

> For fraud detection, **Recall on the Fraud class** is the most important metric — it measures what % of actual frauds were correctly caught.

---

## 💾 Saving the Model

The trained pipeline is saved using `joblib`:

```python
import joblib
joblib.dump(pipeline, "fraud_detection_pipeline.pkl")
```

To load and use later:

```python
pipeline = joblib.load("fraud_detection_pipeline.pkl")
pipeline.predict(new_data)
```

---

## 🌐 Streamlit Web App

The project includes a live prediction web app (`Fraud_DETECTION.py`) built with Streamlit.

### How it works
1. User enters transaction details via input fields
2. App feeds the data into the saved `fraud_detection_pipeline.pkl` model
3. Displays the result instantly with a clear ✅ or ❌ message

### App Inputs

| Field | Description |
|-------|-------------|
| Transaction Type | PAYMENT / TRANSFER / CASH_OUT / DEPOSIT |
| Amount | Transaction amount |
| Old Balance (Sender) | Sender's balance before transaction |
| New Balance (Sender) | Sender's balance after transaction |
| Old Balance (Receiver) | Receiver's balance before transaction |
| New Balance (Receiver) | Receiver's balance after transaction |


---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas & NumPy | Data manipulation |
| Matplotlib & Seaborn | Data visualization |
| Scikit-learn | ML Pipeline, model, evaluation |
| Joblib | Model saving/loading |
| Streamlit | Web app for live predictions |
| Jupyter Notebook | Development environment |

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/fraud-detection.git
cd fraud-detection
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add the dataset**
```
Place "AIML Dataset.csv" in the root folder
```

**4. Run the Jupyter Notebook** (EDA + train the model)
```bash
jupyter notebook analysis_model.ipynb
```

**5. Run the Streamlit Web App** (live predictions)
```bash
streamlit run Fraud_DETECTION.py
```



---

## 📦 requirements.txt

```
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
streamlit
jupyter
```

---

## 👤 Author

**Aditya Mishra**

---

