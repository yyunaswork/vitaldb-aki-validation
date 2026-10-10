# External validation of preoperative AKI risk models in VitalDB

Code for the study "External validation and recalibration of preoperative risk prediction models for acute kidney injury after noncardiac surgery in an open Korean perioperative dataset (VitalDB)".

Author: Mohammad Yunus, MD (Sergiev Posad Hospital, Russia)

## Models validated
Park 2019 (SPARK), Nishimoto 2021, Bell 2015, Nepogodiev 2018. Equations from Zhuo et al., Br J Anaesth 2024; 133: 508-18 (Supplementary Table S10).

## Data
VitalDB open dataset (https://vitaldb.net). Data are downloaded automatically from the public API; no data are stored here.

## How to run (e.g. in Google Colab)
1. Upload both files.
2. Run `%run vitaldb_aki.py` - cohort selection, AKI outcome (KDIGO), AUROC with bootstrap CI, calibration, sensitivity analysis.
3. Run `%run figures.py` - Figures 1-3.

Requirements: Python 3, numpy, pandas, statsmodels, scikit-learn, matplotlib.

## License
MIT
