# Medical Insurance Charges: EDA and Prediction

Which factors drive individual medical insurance charges, and how well can we predict them?
This project explores a 1,338-policyholder dataset, engineers an interaction feature that
lifts a simple linear model to R² ≈ 0.91, and turns the findings into pricing and prevention recommendations.

## Dataset

Public medical insurance dataset (`data/insurance.csv`): 1,338 rows, no missing values.

| Column | Description |
|---|---|
| `age` | Age of the policyholder |
| `sex` | Male / female |
| `bmi` | Body mass index |
| `children` | Number of dependants covered |
| `smoker` | Smoker yes / no |
| `region` | US region (northeast, northwest, southeast, southwest) |
| `charges` | Individual medical charges in USD (**target**) |

## Approach

1. **EDA:** distributions, outliers, correlations, and group comparisons.
2. **Cleaning:** dropped one duplicate row and encoded categorical variables.
3. **Feature engineering:** BMI categories (from the unrounded BMI), an obesity flag, and a **smoker × obese interaction**.
4. **Modeling:** train/test split before scaling (no leakage) with scaling inside the pipeline. Compared linear regression, linear regression on log charges, random forest, and gradient boosting using a held-out test set and 5-fold cross-validation.
5. **Interpretation:** feature importance, an ablation of the interaction feature, and recommendations.

## Key findings

- **Smoking is the biggest driver.** Smokers average about $32.1k in charges against about $8.4k for non-smokers (about 3.8x).
- **Smoking and BMI interact.** Smokers with BMI ≥ 30 average about $41.6k, versus about $21.4k for smokers below 30 and about $8k to $9k for non-smokers at any BMI.
- **Age is the second driver.** Sex, region, and number of children add little.
- **A simple model is enough.** Linear regression with the interaction feature matches the tree ensembles:

| Model | Test R² | Test MAE ($) | CV R² (mean ± std) |
|---|---|---|---|
| Linear Regression | 0.907 | 2,369 | 0.859 ± 0.030 |
| Gradient Boosting | 0.904 | 2,457 | 0.856 ± 0.027 |
| Random Forest | 0.898 | 2,451 | 0.852 ± 0.026 |
| Linear Regression (log target) | 0.674 | 3,845 | 0.443 ± 0.224 |

Without the interaction feature, linear regression drops from R² 0.91 to 0.81.
Log-transforming the target did not help here and is reported as a negative result.

## Recommendations

1. Make smoking status the primary pricing factor.
2. Treat smokers with BMI ≥ 30 as the highest-risk segment for pricing and care management.
3. Direct prevention spend to smoking cessation first, and weight management for smokers.
4. Avoid weighting region or sex heavily; the data shows little signal and it adds fairness and regulatory risk.
5. Prefer an interpretable model: a regression with a well-chosen interaction matches the ensembles.

## Limitations

- Small dataset with no medical history, claims history, or income.
- The dataset is widely used and appears simplified or synthetic, so results may not transfer to real portfolios.
- The largest errors are unusually expensive cases the model underestimates, which points to missing variables such as chronic conditions.

## Next steps

Hyperparameter tuning, a smoker × age interaction, Gamma-GLM or quantile models for skewed costs, and a small Streamlit app for predictions.

