# 🎬 Semi-Supervised Sentiment Analysis with Stacking Ensemble

## Title & Description
This project explores **semi-supervised learning for multiclass sentiment classification** on a movie review dataset containing **60% unlabeled data**.  
We design and compare two end-to-end machine learning pipelines to investigate how different pseudo-labeling strategies and feature representations impact model performance.

### 🔍 Problem
Labeling large-scale text datasets is expensive and time-consuming. This project addresses how to:
- Effectively leverage **unlabeled data** using semi-supervised learning  
- Balance **label quantity vs. label quality**  
- Improve classification performance through **feature engineering and ensemble modeling**

### 🚀 Key Features
- Two complete NLP pipelines:
  1. **Bag-of-Words + Naive Bayes pseudo-labeling + SGD classifier**
  2. **TF-IDF + High-confidence pseudo-labeling + Stacking Ensemble**
- Word-level **and** character-level feature modeling
- Confidence-filtered pseudo-labeling to reduce noise
- Two-level stacking ensemble with multiple base learners
- Final **XGBoost meta-model achieving 94.4% validation accuracy**

---

## Installation

### Prerequisites
- Python **3.9+**
- pip or conda
- (Recommended) Virtual environment

### Install Dependencies
```
pip install -r requirements.txt
```

If installing manually:

```
pip install numpy pandas scikit-learn nltk xgboost
```

### Download NLTK Resources

Run once:

```
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### Data Setup

Place the dataset files in the project root:

```
train.csv
val.csv
test.csv
```

## Usage
Run Full Pipeline

```
python results_notebook.py
```

This executes:
1. Data preprocessing
2. Semi-supervised pseudo-label generation
3. Feature vectorization
4. Model training
5. Validation evaluation

## Pipeline 1: Broad Augmentation Approach

Goal: Use all available unlabeled data.

Steps:
- Heavy preprocessing (stopword removal, lemmatization)
- Bag-of-Words vectorization (20k features)
- Multinomial Naive Bayes predicts missing labels
- SGD classifier trained on augmented dataset

Validation Accuracy: ~92%

## Pipeline 2: High-Confidence Learning Approach (Final Model)

Goal: Prioritize label quality over quantity.

Steps:
- Minimal preprocessing (retain sentiment signals)
- Word-level TF-IDF (1–2 grams)
- Logistic Regression generates pseudo-labels
- Keep only predictions with ≥ 90% confidence
- Train stacking ensemble with:
    - Logistic Regression
    - SGD (modified Huber loss)
    - Complement Naive Bayes
    - Character-level TF-IDF SVM
    - Meta-model: XGBoost classifier

Validation Accuracy: 94.4%

## Example Output

Accuracy: 0.94

Sentiment 0  F1: 0.96
Sentiment 1  F1: 0.90
Sentiment 2  F1: 1.00
Sentiment 3  F1: 0.91

## Author
---
Victoria Piroian

University of Toronto

Faculty of Applied Science & Engineering, 2025
