# 🛒 Market Basket Analysis with K-Means Clustering & Association Rules

This project performs **Market Basket Analysis (MBA)** using both:
- **Association Rule Mining** (Apriori Algorithm)
- **K-Means Clustering** and **PCA Visualization**

to understand and classify customer purchasing behavior for **targeted marketing strategies**.

---

## 👤 Project Information

- **Name**: Upwan Singh  
- **University Roll Number**: 202401100400200  
- **Department**: CSE (AIML)  
- **Institute**: KIET Group of Institutions, Ghaziabad  
- **Session**: 2024–2025  
- **Supervisor**: Abhishek Shukla  

---

## 📁 Dataset

- **File Name**: `10. Market Basket Analysis.csv`
- **Type**: Transactional data (items purchased per transaction)
- **Source**: Internal / academic use

---

## 🎯 Objective

- To identify frequently purchased itemsets using **association rules**.
- To classify customer behavior using **K-Means clustering**.
- To visualize customer segments using **PCA** and **heatmaps**.
- To provide actionable insights for:
  - Product placement
  - Cross-selling
  - Personalized marketing

---

## 🔍 Methods Used

### 1. **Data Preprocessing**
- Cleaned raw transactional data
- Converted rows to list of items per transaction

### 2. **One-Hot Encoding**
- Created binary matrix where:
  - Rows = Transactions
  - Columns = Items
  - Value = 1 if item is present in transaction

### 3. **Association Rule Mining (Optional)**
- Apriori algorithm (can be added for pattern discovery)

### 4. **Clustering (K-Means)**
- Grouped similar transactions using K-Means
- Used `n_clusters=4` (modifiable)

### 5. **Visualization**
- Heatmap: Item frequency per cluster
- PCA: 2D representation of customer clusters

---

## 📊 Visualizations

- ✅ **Heatmap** showing item frequency per customer cluster  
- ✅ **PCA plot** showing distribution of clusters in 2D space

---

## 🧰 Libraries Used

- `pandas`
- `mlxtend`
- `scikit-learn` (KMeans, PCA)
- `seaborn`
- `matplotlib`

---

## 💻 Code Sample

```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df_encoded['Cluster'] = kmeans.fit_predict(df_encoded)

# PCA for visualization
pca = PCA(n_components=2)
components = pca.fit_transform(df_encoded.drop(columns='Cluster'))

# Plot
sns.scatterplot(x=components[:, 0], y=components[:, 1], hue=df_encoded['Cluster'])
plt.title("K-Means Clustering Visualization (PCA)")
