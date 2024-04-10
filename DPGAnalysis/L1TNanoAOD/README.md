# L1T Nano

This package allows to save Level 1 Trigger information (unpacked or reemulated) in a standard NANOAOD format, after reemulating the L1 Trigger (except for the first section, where only unpacked objects are retrieved from MINIAOD). A few examples of cmsDriver commands are provided for Run 3 data and simulated samples. 

 ## Naming conventions
- Unpacked objects are the same as in central NANO and named `L1EG`, `L1Mu`,... 
- Reemulated objects are named `L1EmulEG`, `L1EmulMu`,...
- Unpacked calo TPs/TTs/clusters are named `HcalUnpackedTPs`, `L1UnpackedCaloTower`, `L1UnpackedCaloCluster`,...
- Reemulated calo TPs/TTs/clusters are named `HcalEmulTPs`, `L1EmulCaloTower`, `L1EmulCaloCluster`,...

## L1T NANO from MINIAOD 


### Add L1 objects to central NANO 
On can add all L1 objects on top of central NANO with the following command: 

    cmsDriver.py customL1toNANO --conditions auto:run3_data_prompt -s NANO:@PHYS+@L1FULL --datatier NANOAOD --eventcontent NANOAOD --data --process customl1nano --scenario pp --era Run3 --customise_unsch Configuration/DataProcessing/RecoTLR.customisePostEra_Run3 -n 100 --filein /store/data/Run2023D/Muon1/MINIAOD/22Sep2023_v2-v1/50000/daeae294-4d7c-4f36-8d50-33ef562cbf07.root --fileout file:out.root --python_filename=customl1nano.py 

### Save unpacked objects only (data) 
If one wants to only keep L1 objects, and not offline information, use instead: 

    cmsDriver.py customL1toNANO --conditions auto:run3_data_prompt -s USER:DPGAnalysis/L1TNanoAOD/l1tNano_cff.l1tNanoTask --datatier NANOAOD --eventcontent NANOAOD --data --process customl1nano --scenario pp --era Run3 --customise Configuration/DataProcessing/RecoTLR.customisePostEra_Run3,PhysicsTools/NanoAOD/l1trig_cff.nanoL1TrigObjCustomizeFull -n 100 --filein /store/data/Run2023D/Muon1/MINIAOD/22Sep2023_v2-v1/50000/daeae294-4d7c-4f36-8d50-33ef562cbf07.root --fileout file:out.root --python_filename=customl1nano.py  --fileout file:out.root --python_filename=customl1nano.py 


## L1T NANO standalone
The following examples save L1T information (objects, TPs,...) in a standalone NANOAOD format, after reemulating the L1 trigger. 

### Save all calo information (TPs, TT, CaloCluster) and L1T objects, both emulated and unpacked (data) 

    cmsDriver.py customL1toNANO --conditions auto:run3_data_prompt -s RAW2DIGI,NANO:@L1DPG --datatier NANOAOD --eventcontent NANOAOD --data --process customl1nano --scenario pp --era Run3 --customise Configuration/DataProcessing/RecoTLR.customisePostEra_Run3  -n 100 --filein /store/data/Run2022D/EGamma/RAW-RECO/ZElectron-27Jun2023-v2/2810003/06757985-055e-4c64-bbe3-187858ea2abf.root --fileout file:out.root --python_filename=customl1nano.py   

### Save all calo information (TPs, TT, CaloCluster) and L1T objects, both emulated and unpacked (MC) 

    cmsDriver.py customL1toNANO --conditions auto:phase1_2024_realistic -s RAW2DIGI,NANO:@L1DPG --datatier NANOAOD --eventcontent NANOAOD --mc --process customl1nano --scenario pp --era Run3 --customise Configuration/DataProcessing/RecoTLR.customisePostEra_Run3 -n 100 --filein  /store/mc/Run3Winter24Digi/WJetsToLNu_TuneCP5_13p6TeV-madgraphMLM-pythia8/GEN-SIM-RAW/133X_mcRun3_2024_realistic_v9-v2/2830000/756ef2a9-b60a-4ee1-a021-7219e48c6f4b.root --fileout file:out.root --python_filename=customl1nano.py  
 

## L1T  + central NANO
Also useful is running L1T NANO together with central NANO. Similar commands as above can be defined. A few options exist: one can start from RAW and run the whole RECO/PAT/NANO chain, or start from existing RAW-RECO to skip the time consuming re-RECO step. 
 
 ### Run RECO, PAT, NANO and save all calo information (TPs, TT, CaloCluster) and L1T objects, both emulated and unpacked (data) 

    cmsDriver.py customL1toNANO --conditions auto:run3_data_prompt -s RAW2DIGI,L1Reco,RECO,PAT,NANO:@PHYS+@L1DPG --datatier NANOAOD --eventcontent NANOAOD --data --process customl1nano --scenario pp --era Run3 --customise Configuration/DataProcessing/RecoTLR.customisePostEra_Run3 -n 100 --filein /store/data/Run2023D/EphemeralZeroBias0/RAW/v1/000/370/560/00000/9273062a-1a69-4998-8ae1-c121323526e8.root --fileout file:out.root --python_filename=customl1nano.py    

 
 ### Run RECO, PAT, NANO and save all calo information (TPs, TT, CaloCluster) and L1T objects, both emulated and unpacked (MC) 

    cmsDriver.py customL1toNANO --conditions auto:phase1_2024_realistic -s RAW2DIGI,L1Reco,RECO,PAT,NANO:@PHYS+@L1DPG --datatier NANOAOD --eventcontent NANOAOD --mc --process customl1nano --scenario pp --era Run3 --customise Configuration/DataProcessing/RecoTLR.customisePostEra_Run3 -n 100 --filein /store/mc/Run3Winter24Digi/WJetsToLNu_TuneCP5_13p6TeV-madgraphMLM-pythia8/GEN-SIM-RAW/133X_mcRun3_2024_realistic_v9-v2/2830000/756ef2a9-b60a-4ee1-a021-7219e48c6f4b.root --python_filename=customl1nano.py

 ### Run PAT, NANO and save all calo information (TPs, TT, CaloCluster) and L1T objects, both emulated and unpacked (data) 
 

    cmsDriver.py customL1toNANO --conditions auto:run3_data_prompt -s RAW2DIGI,L1Reco,PAT,NANO:@PHYS+@L1DPG --datatier NANOAOD --eventcontent NANOAOD --data --process customl1nano --scenario pp --era Run3 --customise Configuration/DataProcessing/RecoTLR.customisePostEra_Run3 -n 100 --filein /store/data/Run2022D/EGamma/RAW-RECO/ZElectron-27Jun2023-v2/2810003/06757985-055e-4c64-bbe3-187858ea2abf.root --fileout file:out.root --python_filename=customl1nano.py



## Notes on reemulation options
There exist different options to reemulate the Level 1 Trigger. For calorimeters, one can for example use the original ECAL/HCAL trigger primitives ("unpacked") or reemulate these as well. The former option is the default one. In order to modify this, one can for example add the following customization:

    --customise_unsch L1Trigger/Configuration/customiseReEmul.L1TReEmulFromRAWsimHcalTP
   
Mind that `--customise_unsch` (**and not** `--customise`) is to be used. 



