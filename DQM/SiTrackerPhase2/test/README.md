To run the L1 Tk DQM Plotting script:

Running the DQM plots is split into two parts. You will need to use the GEN-SIM-RECO or GEN-SIM-DIGI-RAW version of the file you want to run on.

To run part 1 of the DQM plotting, ```dqmstep_phase2tk_cfg.py``` is needed. In this code, the user can change the input file. To run code on a small sample size:
```
cmsRun dqmstep_phase2tk_cfg.py 
```

This outputs a file called ```step3_pre4_inDQM.root``` which will be used as input in the next step. To run part 2:
```
cmsRun harvestingstep_phase2tk_cfg.py 
```

This outputs a file called ```DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root``` that allows user to view the histograms locally.