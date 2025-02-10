# Interactive `runTheMatrix` shell

## Introduction

The interactive shell is a place where users can search and explore the many
different workflows that are regularly run via `runTheMatrix`. The interactive
shell is *not* meant to execute any workflow, but simply to explore and
understand all the possibilities made available through the `runTheMatrix`
script.

To enter the `runTheMatrix` interactive shell, issue the command:

```
runTheMatrix.py --interactive
```

If the `--interactive` option is specified at run-time, it will take precedence
over all other specified options and workflows that, therefore, will not be
executed.

To exit the shell, use the command `q` or `Ctrl-D`.

To display the help menu, use the command `?`.


## Available commands

All commands have an accompanying help that is meant to illustrate the correct
syntax to be used to run them. A brief explanation is also provided.
To show the help for a specific command, issue the command: `help
specific_command`.

If, for example, at the prompt, you type:
```
matrix> help search
search search_regexp

This command will search for a match within all workflows registered.
The search is done on both the workflow name and the names of steps registered to it.
matrix>
```

You will get an explanation of the command together with its correct syntax.

### Generic Search

This is the most inclusive search available and it could be useful to search for
a specific workflow among *all* workflows.

For example, if you want to know all workflows that use the geometry `D49` and
generate `SingleElectron`, you could use something like:

```
matrix> search .*D49.*SingleElectron.*
Found 0 compatible workflows inside relval_gpu
Found 0 compatible workflows inside relval_production
Found 0 compatible workflows inside relval_identity
Found 0 compatible workflows inside relval_ged
Found 0 compatible workflows inside relval_highstats
Found 0 compatible workflows inside relval_generator
Found 0 compatible workflows inside relval_standard
Found 0 compatible workflows inside relval_extendedgen
Found 0 compatible workflows inside relval_premix
Found 0 compatible workflows inside relval_Run4
Found 0 compatible workflows inside relval_machine
Found 0 compatible workflows inside relval_pileup
Found 0 compatible workflows inside relval_2017
23201.0 SingleElectronPt10 Run4D49+SingleElectronPt10_pythia8_GenSimHLBeamSpot+DigiTrigger+RecoGlobal+HARVESTGlobal
23202.0 SingleElectronPt35 Run4D49+SingleElectronPt35_pythia8_GenSimHLBeamSpot+DigiTrigger+RecoGlobal+HARVESTGlobal
23203.0 SingleElectronPt1000 Run4D49+SingleElectronPt1000_pythia8_GenSimHLBeamSpot+DigiTrigger+RecoGlobal+HARVESTGlobal
23291.0 SingleElectronPt15Eta1p7_2p7 Run4D49+SingleElectronPt15Eta1p7_2p7_GenSimHLBeamSpot+DigiTrigger+RecoGlobal+HARVESTGlobal
23302.0 SingleEFlatPt2To100 Run4D49+SingleElectronFlatPt2To100_GenSimHLBeamSpot+DigiTrigger+RecoGlobal+HARVESTGlobal
23401.0 SingleElectronPt10 Run4D49PU+SingleElectronPt10_pythia8_GenSimHLBeamSpot+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23401.98 SingleElectronPt10 Run4D49PU_PMXS2+SingleElectronPt10_pythia8_GenSimHLBeamSpot+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23401.99 SingleElectronPt10 Run4D49PU_PMXS1S2+SingleElectronPt10_pythia8_GenSimHLBeamSpot+PREMIX_PremixHLBeamSpotPU+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23401.999 SingleElectronPt10 Run4D49PU_PMXS1S2PR+SingleElectronPt10_pythia8_GenSimHLBeamSpot+PREMIX_PremixHLBeamSpotPU+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23402.0 SingleElectronPt35 Run4D49PU+SingleElectronPt35_pythia8_GenSimHLBeamSpot+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23402.98 SingleElectronPt35 Run4D49PU_PMXS2+SingleElectronPt35_pythia8_GenSimHLBeamSpot+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23402.99 SingleElectronPt35 Run4D49PU_PMXS1S2+SingleElectronPt35_pythia8_GenSimHLBeamSpot+PREMIX_PremixHLBeamSpotPU+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23402.999 SingleElectronPt35 Run4D49PU_PMXS1S2PR+SingleElectronPt35_pythia8_GenSimHLBeamSpot+PREMIX_PremixHLBeamSpotPU+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23403.0 SingleElectronPt1000 Run4D49PU+SingleElectronPt1000_pythia8_GenSimHLBeamSpot+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23403.98 SingleElectronPt1000 Run4D49PU_PMXS2+SingleElectronPt1000_pythia8_GenSimHLBeamSpot+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23403.99 SingleElectronPt1000 Run4D49PU_PMXS1S2+SingleElectronPt1000_pythia8_GenSimHLBeamSpot+PREMIX_PremixHLBeamSpotPU+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23403.999 SingleElectronPt1000 Run4D49PU_PMXS1S2PR+SingleElectronPt1000_pythia8_GenSimHLBeamSpot+PREMIX_PremixHLBeamSpotPU+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23491.0 SingleElectronPt15Eta1p7_2p7 Run4D49PU+SingleElectronPt15Eta1p7_2p7_GenSimHLBeamSpot+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23491.98 SingleElectronPt15Eta1p7_2p7 Run4D49PU_PMXS2+SingleElectronPt15Eta1p7_2p7_GenSimHLBeamSpot+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23491.99 SingleElectronPt15Eta1p7_2p7 Run4D49PU_PMXS1S2+SingleElectronPt15Eta1p7_2p7_GenSimHLBeamSpot+PREMIX_PremixHLBeamSpotPU+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23491.999 SingleElectronPt15Eta1p7_2p7 Run4D49PU_PMXS1S2PR+SingleElectronPt15Eta1p7_2p7_GenSimHLBeamSpot+PREMIX_PremixHLBeamSpotPU+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23502.0 SingleEFlatPt2To100 Run4D49PU+SingleElectronFlatPt2To100_GenSimHLBeamSpot+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23502.98 SingleEFlatPt2To100 Run4D49PU_PMXS2+SingleElectronFlatPt2To100_GenSimHLBeamSpot+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23502.99 SingleEFlatPt2To100 Run4D49PU_PMXS1S2+SingleElectronFlatPt2To100_GenSimHLBeamSpot+PREMIX_PremixHLBeamSpotPU+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23502.999 SingleEFlatPt2To100 Run4D49PU_PMXS1S2PR+SingleElectronFlatPt2To100_GenSimHLBeamSpot+PREMIX_PremixHLBeamSpotPU+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
Found 25 compatible workflows inside relval_upgrade
matrix>
```

The search terms in this and in all other commands should be *valid regular
expressions*. Those are the most powerful tool for this kind of job.
Maybe not the most intuitive one, if you are not too familiar with them.

### Search in a specific set of workflows

If, instead, you would like to limit the search to a specific _macro area_, e.g.
relval_upgrade, you could use the same regular expression but a different
command:

```
matrix> help searchInWorkflow
searchInWorkflow wfl_name search_regexp

This command will search for a match within all workflows registered to wfl_name.
The search is done on both the workflow name and the names of steps registered to it.
```

Example:

```
matrix> searchInWorkflow relval_upgrade .*D49.*SingleElectron.*
23201.0 SingleElectronPt10 Run4D49+SingleElectronPt10_pythia8_GenSimHLBeamSpot+DigiTrigger+RecoGlobal+HARVESTGlobal
23202.0 SingleElectronPt35 Run4D49+SingleElectronPt35_pythia8_GenSimHLBeamSpot+DigiTrigger+RecoGlobal+HARVESTGlobal
23203.0 SingleElectronPt1000 Run4D49+SingleElectronPt1000_pythia8_GenSimHLBeamSpot+DigiTrigger+RecoGlobal+HARVESTGlobal
23291.0 SingleElectronPt15Eta1p7_2p7 Run4D49+SingleElectronPt15Eta1p7_2p7_GenSimHLBeamSpot+DigiTrigger+RecoGlobal+HARVESTGlobal
23302.0 SingleEFlatPt2To100 Run4D49+SingleElectronFlatPt2To100_GenSimHLBeamSpot+DigiTrigger+RecoGlobal+HARVESTGlobal
23401.0 SingleElectronPt10 Run4D49PU+SingleElectronPt10_pythia8_GenSimHLBeamSpot+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23401.98 SingleElectronPt10 Run4D49PU_PMXS2+SingleElectronPt10_pythia8_GenSimHLBeamSpot+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23401.99 SingleElectronPt10 Run4D49PU_PMXS1S2+SingleElectronPt10_pythia8_GenSimHLBeamSpot+PREMIX_PremixHLBeamSpotPU+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23401.999 SingleElectronPt10 Run4D49PU_PMXS1S2PR+SingleElectronPt10_pythia8_GenSimHLBeamSpot+PREMIX_PremixHLBeamSpotPU+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23402.0 SingleElectronPt35 Run4D49PU+SingleElectronPt35_pythia8_GenSimHLBeamSpot+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23402.98 SingleElectronPt35 Run4D49PU_PMXS2+SingleElectronPt35_pythia8_GenSimHLBeamSpot+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23402.99 SingleElectronPt35 Run4D49PU_PMXS1S2+SingleElectronPt35_pythia8_GenSimHLBeamSpot+PREMIX_PremixHLBeamSpotPU+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23402.999 SingleElectronPt35 Run4D49PU_PMXS1S2PR+SingleElectronPt35_pythia8_GenSimHLBeamSpot+PREMIX_PremixHLBeamSpotPU+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23403.0 SingleElectronPt1000 Run4D49PU+SingleElectronPt1000_pythia8_GenSimHLBeamSpot+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23403.98 SingleElectronPt1000 Run4D49PU_PMXS2+SingleElectronPt1000_pythia8_GenSimHLBeamSpot+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23403.99 SingleElectronPt1000 Run4D49PU_PMXS1S2+SingleElectronPt1000_pythia8_GenSimHLBeamSpot+PREMIX_PremixHLBeamSpotPU+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23403.999 SingleElectronPt1000 Run4D49PU_PMXS1S2PR+SingleElectronPt1000_pythia8_GenSimHLBeamSpot+PREMIX_PremixHLBeamSpotPU+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23491.0 SingleElectronPt15Eta1p7_2p7 Run4D49PU+SingleElectronPt15Eta1p7_2p7_GenSimHLBeamSpot+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23491.98 SingleElectronPt15Eta1p7_2p7 Run4D49PU_PMXS2+SingleElectronPt15Eta1p7_2p7_GenSimHLBeamSpot+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23491.99 SingleElectronPt15Eta1p7_2p7 Run4D49PU_PMXS1S2+SingleElectronPt15Eta1p7_2p7_GenSimHLBeamSpot+PREMIX_PremixHLBeamSpotPU+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23491.999 SingleElectronPt15Eta1p7_2p7 Run4D49PU_PMXS1S2PR+SingleElectronPt15Eta1p7_2p7_GenSimHLBeamSpot+PREMIX_PremixHLBeamSpotPU+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23502.0 SingleEFlatPt2To100 Run4D49PU+SingleElectronFlatPt2To100_GenSimHLBeamSpot+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23502.98 SingleEFlatPt2To100 Run4D49PU_PMXS2+SingleElectronFlatPt2To100_GenSimHLBeamSpot+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23502.99 SingleEFlatPt2To100 Run4D49PU_PMXS1S2+SingleElectronFlatPt2To100_GenSimHLBeamSpot+PREMIX_PremixHLBeamSpotPU+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23502.999 SingleEFlatPt2To100 Run4D49PU_PMXS1S2PR+SingleElectronFlatPt2To100_GenSimHLBeamSpot+PREMIX_PremixHLBeamSpotPU+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
Found 25 compatible workflows inside relval_upgrade
matrix>
```

### Explore/Dump a specific workflow

Suppose now that you want to know what are the commands run by the workflow
`23291`. You could do by doing:

```
matrix> dumpWorkflowId 23291
23291.0 Run4D49+SingleElectronPt15Eta1p7_2p7_GenSimHLBeamSpot+DigiTrigger+RecoGlobal+HARVESTGlobal
[1]: cmsDriver.py SingleElectronPt15Eta1p7_2p7_cfi  --conditions auto:phase2_realistic_T15 -n 10 --era Phase2C9 --eventcontent FEVTDEBUG --relval 9000,100 -s GEN,SIM --datatier GEN-SIM --beamspot HLLHC --geometry ExtendedRun4D49

[2]: cmsDriver.py step2  --conditions auto:phase2_realistic_T15 -s DIGI:pdigi_valid,L1TrackTrigger,L1,DIGI2RAW,HLT:@fake2 --datatier GEN-SIM-DIGI-RAW -n 10 --geometry ExtendedRun4D49 --era Phase2C9 --eventcontent FEVTDEBUGHLT

[3]: cmsDriver.py step3  --conditions auto:phase2_realistic_T15 -s RAW2DIGI,L1Reco,RECO,RECOSIM,PAT,VALIDATION:@phase2Validation+@miniAODValidation,DQM:@phase2+@miniAODDQM --datatier GEN-SIM-RECO,MINIAODSIM,DQMIO -n 10 --geometry ExtendedRun4D49 --era Phase2C9 --eventcontent FEVTDEBUGHLT,MINIAODSIM,DQM

[4]: cmsDriver.py step4  --conditions auto:phase2_realistic_T15 -s HARVESTING:@phase2Validation+@phase2+@miniAODValidation+@miniAODDQM --scenario pp --filetype DQM --geometry ExtendedRun4D49 --era Phase2C9 --mc  -n 100


Workflow found in relval_upgrade.
matrix>
```


### Predefined set of workflows

The `predefined` command, instead, could be used to know what is the list of
predefined workflows that have been bundled together and, for each of them,
query the list of registered workflow numbers. E.g.:

```
matrix> help predefined
predefined [predef1 [...]]

Run w/o argument, it will print the list of known predefined workflows.
Run with space-separated predefined workflows, it will print the workflow-ids registered to them
```

Example:
```
matrix> predefined
List of predefined workflows
['jetmc', 'limited', 'muonmc', 'metmc']
matrix> predefined jetmc
List of predefined workflows
Predefined Set: jetmc
[5.1, 13, 15, 25, 38, 39]
matrix>
```

### Explore "macro-workflows"

The last command is `showWorkflow`, that does the following:

```
matrix> help showWorkflow
showWorkflow [workflow1 [...]]

Run w/o arguments, it will print the list of registered macro-workflows.
Run with space-separated workflows, it will print the full list of workflow-ids registered to them
```

Example:
```
matrix> showWorkflow
Available workflows:
standard
highstats
pileup
generator
extendedgen
production
ged
upgrade
cleanedupgrade
gpu
2017
Run4
identity
machine
premix
nano

matrix> showWorkflow gpu
141.008506 Run3-2023_JetMET2023B_RecoPixelOnlyTripletsGPU
141.008507 Run3-2023_JetMET2023B_RecoPixelOnlyTripletsGPU_Validation
141.008508 Run3-2023_JetMET2023B_RecoPixelOnlyTripletsGPU_Profiling
141.008512 Run3-2023_JetMET2023B_RecoECALOnlyGPU
141.008513 Run3-2023_JetMET2023B_RecoECALOnlyGPU_Validation
141.008514 Run3-2023_JetMET2023B_RecoECALOnlyGPU_Profiling
141.008522 Run3-2023_JetMET2023B_RecoHCALOnlyGPU
141.008523 Run3-2023_JetMET2023B_RecoHCALOnlyGPU_Validation
141.008524 Run3-2023_JetMET2023B_RecoHCALOnlyGPU_Profiling
141.008583 Run3-2023_JetMET2023B_GPUValidation
160.03502 HydjetQ_MinBias_5362GeV_2023_ppReco
12450.406 ZMM_14+2023_Patatrack_PixelOnlyTripletsAlpaka
12450.407 ZMM_14+2023_Patatrack_PixelOnlyTripletsAlpaka_Validation
12450.408 ZMM_14+2023_Patatrack_PixelOnlyTripletsAlpaka_Profiling
12834.402 TTbar_14TeV+2024_Patatrack_PixelOnlyAlpaka
12834.403 TTbar_14TeV+2024_Patatrack_PixelOnlyAlpaka_Validation
12834.404 TTbar_14TeV+2024_Patatrack_PixelOnlyAlpaka_Profiling
12834.406 TTbar_14TeV+2024_Patatrack_PixelOnlyTripletsAlpaka
12834.407 TTbar_14TeV+2024_Patatrack_PixelOnlyTripletsAlpaka_Validation
12834.408 TTbar_14TeV+2024_Patatrack_PixelOnlyTripletsAlpaka_Profiling
12834.412 TTbar_14TeV+2024_Patatrack_ECALOnlyAlpaka
12834.422 TTbar_14TeV+2024_Patatrack_HCALOnlyAlpaka_Validation
12834.423 TTbar_14TeV+2024_Patatrack_HCALOnlyGPUandAlpaka_Validation
12834.424 TTbar_14TeV+2024_Patatrack_HCALOnlyAlpaka_Profiling
12850.402 ZMM_14+2024_Patatrack_PixelOnlyAlpaka
12850.403 ZMM_14+2024_Patatrack_PixelOnlyAlpaka_Validation
12850.404 ZMM_14+2024_Patatrack_PixelOnlyAlpaka_Profiling
12861.402 NuGun+2024_Patatrack_PixelOnlyAlpaka
13034.402 TTbar_14TeV+2024PU_Patatrack_PixelOnlyAlpaka
13034.403 TTbar_14TeV+2024PU_Patatrack_PixelOnlyAlpaka_Validation
13034.404 TTbar_14TeV+2024PU_Patatrack_PixelOnlyAlpaka_Profiling
13034.406 TTbar_14TeV+2024PU_Patatrack_PixelOnlyTripletsAlpaka
13034.407 TTbar_14TeV+2024PU_Patatrack_PixelOnlyTripletsAlpaka_Validation
13034.408 TTbar_14TeV+2024PU_Patatrack_PixelOnlyTripletsAlpaka_Profiling
13034.412 TTbar_14TeV+2024PU_Patatrack_ECALOnlyAlpaka
13034.422 TTbar_14TeV+2024PU_Patatrack_HCALOnlyAlpaka_Validation
13034.423 TTbar_14TeV+2024PU_Patatrack_HCALOnlyGPUandAlpaka_Validation
13034.424 TTbar_14TeV+2024PU_Patatrack_HCALOnlyAlpaka_Profiling
13050.402 ZMM_14+2024PU_Patatrack_PixelOnlyAlpaka
13050.403 ZMM_14+2024PU_Patatrack_PixelOnlyAlpaka_Validation
13050.404 ZMM_14+2024PU_Patatrack_PixelOnlyAlpaka_Profiling
13050.406 ZMM_14+2024PU_Patatrack_PixelOnlyTripletsAlpaka
13050.407 ZMM_14+2024PU_Patatrack_PixelOnlyTripletsAlpaka_Validation
13050.408 ZMM_14+2024PU_Patatrack_PixelOnlyTripletsAlpaka_Profiling
13061.402 NuGun+2024PU_Patatrack_PixelOnlyAlpaka
24834.704 TTbar_14TeV+Run4D98_lstOnGPUIters01TrackingOnly
29634.402 TTbar_14TeV+Run4D110_Patatrack_PixelOnlyAlpaka
29634.403 TTbar_14TeV+Run4D110_Patatrack_PixelOnlyAlpaka_Validation
29634.404 TTbar_14TeV+Run4D110_Patatrack_PixelOnlyAlpaka_Profiling
29634.406 TTbar_14TeV+Run4D110_Patatrack_PixelOnlyTripletsAlpaka
29661.402 NuGun+Run4D110_Patatrack_PixelOnlyAlpaka
29834.402 TTbar_14TeV+Run4D110PU_Patatrack_PixelOnlyAlpaka
29834.403 TTbar_14TeV+Run4D110PU_Patatrack_PixelOnlyAlpaka_Validation
29834.404 TTbar_14TeV+Run4D110PU_Patatrack_PixelOnlyAlpaka_Profiling
gpu contains 54 workflows
matrix> 
```

All commands come with dynamic TAB-completion. There's also a transient history
of the commands issued within a single session. Transient means that, after a
session is closed, the history is lost.

### Limited Matrix for (also) PR Testing

The "limited" predefined set of workflows is used in PR integration testing. Here the workflows run.

MC workflows for pp collisions:

| **WF** 	| **Fragment/Input** 	| **Conditions** 	| **Era** 	| **Notes** 	|  	
|---	|---	|---	|---	|---	|	
| |  	|  	|  	|  	|  	
| **Run1** 	|  	|  	|  	|  	|  	
| |  	|  	|  	|  	|  	
| 5.1 	|  	TTbar_8TeV_TuneCUETP8M1 | run1_mc 	|  	| *FastSim* 	|  	
| 8 	| RelValBeamHalo 	| run1_mc 	|  	| Cosmics 	|  	
| 9.0 	| RelValHiggs200ChargedTaus 	| run1_mc 	|  	|  	|  	
| 25 	| RelValTTbar 	| run1_mc 	|  	|  	|  	
| 101.0 	| SingleElectronE120EHCAL 	| run1_mc 	|  	| + ECALHCAL.customise + fullMixCustomize_cff.setCrossingFrameOn 	|  	
| |  	|  	|  	|  	|  	
| **Run2** 	|  	|  	|  	|  	|  	
| |  	|  	|  	|  	|  	
| 7.3 	| UndergroundCosmicSPLooseMu 	| run2_2018 	|  	|  	|  	
| 1306.0 	| RelValSingleMuPt1_UP15 	| run2_mc 	| Run2_2016 	| with miniAOD 	|  	
| 1330 	| RelValZMM_13 	| run2_mc 	| Run2_2016 	|  	|  	
| 135.4 	| ZEE_13TeV_TuneCUETP8M1 	| run2_mc 	| Run2_2016 	| *FastSim* 	|  	
| 25202.0 	| RelValTTbar_13 	| run2_mc 	| Run2_2016 	|  AVE_35_BX_25ns 	|  	
| 250202.181 | RelValTTbar_13 (PREMIX) 	| phase1_2018_realistic 	| Run2_2018 	|  	|  	|  
| |  	|  	|  	|  	|  	
| **Run3** 	|  	|  	|  	|  	|  	
| |  	|  	|  	|  	|  	
| 11634.0 	| TTbar_14TeV 	| phase1_2022_realistic 	| Run3 	|  	|  	
| 13234.0 	| RelValTTbar_14TeV 	| phase1_2022_realistic 	| Run3_FastSim 	| *FastSim*  |  	
| 12434.0 	| RelValTTbar_14TeV 	| phase1_2023_realistic 	| Run3_2023 	|  	|  	
| 12834.0 	| RelValTTbar_14TeV 	| phase1_2024_realistic 	| Run3_2024 	|  	| 
| 12846.0 	| RelValZEE_14 	| phase1_2023_realistic 	| Run3_2024 	|  	|  	
| 13034.0 	| RelValTTbar_14TeV 	| phase1_2024_realistic 	| Run3_2024 	|  Run3_Flat55To75_PoissonOOTPU 	|  	
| 12834.7 	| RelValTTbar_14TeV 	| phase1_2024_realistic 	| Run3_2024 	| mkFit 	|  	
| 12834.0 	| RelValTTbar_14TeV 	| phase1_2024_realistic 	| Run3_2024 	|  	| 
| 16834.0 	| RelValTTbar_14TeV 	| phase1_2025_realistic 	| Run3_2025 	|  	| 
| 14034.0 	| RelValTTbar_14TeV 	| phase1_2023_realistic 	| Run3_2023_FastSim 	|  	*FastSim* |  	
| 14234.0 	| RelValTTbar_14TeV 	| phase1_2023_realistic 	| Run3_2023_FastSim 	| *FastSim*  Run3_Flat55To75_PoissonOOTPU 	|  	
| 2500.201 	| RelValTTbar_14TeV 	| phase1_2022_realistic 	| Run3 	| NanoAOD from existing MINI 	|  	
| | | | | | 
| **Phase2** 	|  	|  	|  	|  	**Geometry** |  	
| |  	|  	|  	|  	|  	
| 24834.0 	| RelValTTbar_14TeV 	| phase2_realistic_T25 	| Phase2C17I13M9 	| ExtendedRun4D98 	| (Phase-2 baseline) 	
| 24834.911 	| TTbar_14TeV_TuneCP5 	| phase2_realistic_T25 	| Phase2C17I13M9 	| DD4hepExtendedRun4D98 	| DD4Hep (HLLHC14TeV BeamSpot) 	
| 25034.999 	| RelValTTbar_14TeV (PREMIX) 	| phase2_realistic_T25 	| Phase2C17I13M9 	| ExtendedRun4D98 	| AVE_50_BX_25ns_m3p3 	
| 24896.0 	| RelValCloseByPGun_CE_E_Front_120um 	| phase2_realistic_T25 	| Phase2C17I13M9 	| ExtendedRun4D98 	|  	
| 24900.0 	| RelValCloseByPGun_CE_H_Coarse_Scint 	| phase2_realistic_T25 	| Phase2C17I13M9 	| ExtendedRun4D98 	|  	
| 23234.0 	| TTbar_14TeV_TuneCP5 	| phase2_realistic_T21 	| Phase2C20I13M9 	| ExtendedRun4D94 	| (exercise with HFNose) 	

pp Data reRECO workflows:

| Data 	|  	|  	|  	|  	|  	
|---	|---	|---	|---	|---	|	
| **WF** 	| **Input** 	| **Conditions** 	| **Era** 	| **Notes** 	|  	
| |  	|  	|  	|  	|  	
| **Run1** 	|  	|  	|  	|  	|  	
| |  	|  	|  	|  	|  	
| 4.22 	| Run2011A Cosmics 	|  	run1_data |  	|  	*Cosmics* |
| 4.53 	| Run2012B Photon 	| run1_hlt_Fake 	| | + miniAODs 	|  	
| 1000 	| Run2011A MinimumBias Prompt 	| run1_data 	| | + RecoTLR.customisePrompt 	|  	
| 1001 	| Run2011A 	MinimumBias  |  	run1_data |  	| Data+Express | 	
| |  	|  	|  	|  	|  	
| **Run2** 	|  	|  	|  	|  	|  	
| |  	|  	|  	|  	|  	
| 136.731 	| Run2016B SinglePhoton 	|  	|  	|  	|  	
| 136.7611 	| Run2016E JetHT (reMINIAOD) 	| run2_data 	| Run2_2016_HIPM 	| + run2_miniAOD_80XLegacy 	custom |  	
| 136.8311 	| Run2017F JetHT (reMINIAOD) 	| run2_data 	| Run2_2017 	| + run2_miniAOD_94XFall17 	custom |  	
| 136.88811 	| Run2018D JetHT (reMINIAOD) 	| run2_data 	| Run2_2018 	| + run2_miniAOD_UL_preSummer20 (UL MINI) custom |  	
| 136.793 	| Run2017C DoubleEG 	| run2_hlt_relval 	| Run2_2017 	| HLT:@relval2017|  	
| 136.874 	| Run2018C EGamma 	| run2_hlt_relval 	| Run2_2018 	| HLT@relval2018 	|  	
| |  	|  	|  	|  	|  	
| **Run3** 	|  	|  	|  	|  	|  	
| |  	|  	|  	|  	|  	
| 2021 	|  	|  	|  	|  	|  	
| 139.001 	| Run2021 	MinimumBias 	| run3_hlt_relval 	| Run3 	| HLT@relval2022 (Commissioning2021) |	
| 2022 	|  	|  	|  	|  	|  	
| 2022.002001 	| Run2022D ZeroBias 	|  	run3_hlt_relval + run3_data_relval |  	Run3 |  	HLT:@relval2022 | 
| 2022.000001 	| Run2022D JetHT 	|  	run3_hlt_relval + run3_data_relval |  	Run3 |  	HLT:@relval2022 |  	
| 2023 	|  	|  	|  	|  	|  
| 2023.002001 	| Run2023D ZeroBias 	|  	run3_hlt_relval + run3_data_relval|  	Run3_2023 |  	HLT:@relval2023 | 
| 2023.000001 	| Run2023D MuonEG 	|  	run3_hlt_relval + run3_data_relval|  	Run3_2023 |  	HLT:@relval2023 | 
| 2024 	|  	|  	|  	|  	|
| 145.014 	| Run2024B ZeroBias 	|  	run3_hlt_relval + run3_data_prompt_relval|  	Run3_2024 |  	HLT:@relval2025 |  
| 145.104 	| Run2024C JetMet0 	|  	run3_hlt_relval + run3_data_prompt_relval|  	Run3_2024 |  	HLT:@relval2025 |  
| 145.202 	| Run2024D EGamma0 	|  	run3_hlt_relval + run3_data_prompt_relval|  	Run3_2024 |  	HLT:@relval2025 |  
| 145.301 	| Run2024E DisplacedJet 	|  	run3_hlt_relval + run3_data_prompt_relval|  	Run3_2024 |  	HLT:@relval2025 |  
| 145.408 	| Run2024B ParkingDoubleMuonLowMass0 	|  	run3_hlt_relval + run3_data_prompt_relval|  	Run3_2024 |  	HLT:@relval2025 |  
| 145.500 	| Run2024B BTagMu 	|  	run3_hlt_relval + run3_data_prompt_relval|  	Run3_2024 |  	HLT:@relval2025 |  
| 145.604 	| Run2024B JetMET0 	|  	run3_hlt_relval + run3_data_prompt_relval|  	Run3_2024 |  	HLT:@relval2025 |  
| 145.713 	| Run2024B Tau 	|  	run3_hlt_relval + run3_data_prompt_relval|  	Run3_2024 |  	HLT:@relval2025 |  

And Heavy Ion workflows:

| **HIon** 	|  	|  	|  	|  	|  	
|---	|---	|---	|---	|---	|	
| **WF** 	| **Fragment/Input** 	| **Conditions** 	| **Era** 	| **Notes** 	| 
|  	|  	|  	|  	|  	| 	
| **Data** 	|  	|  	|  	|  	|  
|  	|  	|  	|  	|  	|	
| 140.53 	| HIRun2011 HIMinBiasUPC 	| run1_data 	|  	|  	  	
| 140.56 	| HIRun2018A HIHardProbes 	| run2_data_promptlike_hi 	| Run2_2018_pp_on_AA 	|  	  	
|  	|  	|  	|  	|  	|
| **MC** 	|  	|  	|  	|  	|  
|  	|  	|  	|  	|  	|  	
| 158.01 	| RelValHydjetQ_B12_5020GeV_2018_ppReco (reMINIAOD) | phase1_2018_realistic_hi 	| Run2_2018_pp_on_AA 	| (HI MC with pp-like reco) 	|  	  	
| 312.0 	| Pyquen_ZeemumuJets_pt10_2760GeV 	|  phase1_2022_realistic_hi	| Run3_pp_on_PbPb 	| PU = HiMixGEN 	|  
| 322.0 	| Pyquen_ZeemumuJets_pt10_5362GeV_2023 	|  phase1_2023_realistic_hi	| Run3_pp_on_PbPb_2023 	| PU = HiMixGEN 	|  
| 332.0 	| Pyquen_ZeemumuJets_pt10_5362GeV_2024 	|  phase1_2024_realistic_hi	| Run3_pp_on_PbPb_2024 	| PU = HiMixGEN 	|  
