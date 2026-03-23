# Phase2 Tracker DQM:

Producing DQM plots is split into two parts. 

Input file is GEN-SIM-RECO or GEN-SIM-DIGI-RAW .root file, inside ```dqmstep_phase2tk_cfg.py```.

Step 1 of the DQM plotting: 
```
cmsRun dqmstep_phase2tk_cfg.py
```

The output file ```step3_pre4_inDQM.root``` is then used as input in the step 2 (harvesting):
```
cmsRun harvestingstep_phase2tk_cfg.py
```

The final output is ```DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root``` with DQM histograms.

# Phase2 C-RACK DQM:

C-RACK is Cosmic Rack test stand in TIF with up to 6 Ladders of 12 2S modules.

To produce DQM plots on C-RACK (for MC, tested, and eventually data, yet to be tested) there are dedicated scripts for both DQM and Harvesting steps.

Step 1:
```
cmsRun dqmstep_phase2c-rack_cfg.py
```
Step 2:
```
cmsRun harvestingstep_phase2c-rack_cfg.py
```
These C-RACK scripts include D500 geometry, while not including Inner Tracker steps. RecHit (tracking part not yet defined) and Validation steps are commented.

CRACK DQM steps are defined into ```python/Phase2CRackDQMFirstStep_cff.py```.

DQM plots to be produced only for C-RACK could be set with ```switch = false```, and enabled inside dedicated .cff, like ```python/Phase2CRackMonitorCluster_cff.py```.