cp Empty_Sqlite.db Gains_Sqlite.db
cmsRun Gains_Compute_cfg.py
root -l -b -q KeepOnlyGain.C+
cmsRun Validation_Compute_cfg.py
root -l -b -q PlotMacro.C+
