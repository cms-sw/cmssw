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
Found 0 compatible workflows inside relval_2026
Found 0 compatible workflows inside relval_machine
Found 0 compatible workflows inside relval_pileup
Found 0 compatible workflows inside relval_2017
23201.0 SingleElectronPt10 2026D49+SingleElectronPt10_pythia8_GenSimHLBeamSpot+DigiTrigger+RecoGlobal+HARVESTGlobal
23202.0 SingleElectronPt35 2026D49+SingleElectronPt35_pythia8_GenSimHLBeamSpot+DigiTrigger+RecoGlobal+HARVESTGlobal
23203.0 SingleElectronPt1000 2026D49+SingleElectronPt1000_pythia8_GenSimHLBeamSpot+DigiTrigger+RecoGlobal+HARVESTGlobal
23291.0 SingleElectronPt15Eta1p7_2p7 2026D49+SingleElectronPt15Eta1p7_2p7_GenSimHLBeamSpot+DigiTrigger+RecoGlobal+HARVESTGlobal
23302.0 SingleEFlatPt2To100 2026D49+SingleElectronFlatPt2To100_GenSimHLBeamSpot+DigiTrigger+RecoGlobal+HARVESTGlobal
23401.0 SingleElectronPt10 2026D49PU+SingleElectronPt10_pythia8_GenSimHLBeamSpot+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23401.98 SingleElectronPt10 2026D49PU_PMXS2+SingleElectronPt10_pythia8_GenSimHLBeamSpot+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23401.99 SingleElectronPt10 2026D49PU_PMXS1S2+SingleElectronPt10_pythia8_GenSimHLBeamSpot+PREMIX_PremixHLBeamSpotPU+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23401.999 SingleElectronPt10 2026D49PU_PMXS1S2PR+SingleElectronPt10_pythia8_GenSimHLBeamSpot+PREMIX_PremixHLBeamSpotPU+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23402.0 SingleElectronPt35 2026D49PU+SingleElectronPt35_pythia8_GenSimHLBeamSpot+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23402.98 SingleElectronPt35 2026D49PU_PMXS2+SingleElectronPt35_pythia8_GenSimHLBeamSpot+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23402.99 SingleElectronPt35 2026D49PU_PMXS1S2+SingleElectronPt35_pythia8_GenSimHLBeamSpot+PREMIX_PremixHLBeamSpotPU+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23402.999 SingleElectronPt35 2026D49PU_PMXS1S2PR+SingleElectronPt35_pythia8_GenSimHLBeamSpot+PREMIX_PremixHLBeamSpotPU+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23403.0 SingleElectronPt1000 2026D49PU+SingleElectronPt1000_pythia8_GenSimHLBeamSpot+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23403.98 SingleElectronPt1000 2026D49PU_PMXS2+SingleElectronPt1000_pythia8_GenSimHLBeamSpot+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23403.99 SingleElectronPt1000 2026D49PU_PMXS1S2+SingleElectronPt1000_pythia8_GenSimHLBeamSpot+PREMIX_PremixHLBeamSpotPU+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23403.999 SingleElectronPt1000 2026D49PU_PMXS1S2PR+SingleElectronPt1000_pythia8_GenSimHLBeamSpot+PREMIX_PremixHLBeamSpotPU+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23491.0 SingleElectronPt15Eta1p7_2p7 2026D49PU+SingleElectronPt15Eta1p7_2p7_GenSimHLBeamSpot+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23491.98 SingleElectronPt15Eta1p7_2p7 2026D49PU_PMXS2+SingleElectronPt15Eta1p7_2p7_GenSimHLBeamSpot+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23491.99 SingleElectronPt15Eta1p7_2p7 2026D49PU_PMXS1S2+SingleElectronPt15Eta1p7_2p7_GenSimHLBeamSpot+PREMIX_PremixHLBeamSpotPU+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23491.999 SingleElectronPt15Eta1p7_2p7 2026D49PU_PMXS1S2PR+SingleElectronPt15Eta1p7_2p7_GenSimHLBeamSpot+PREMIX_PremixHLBeamSpotPU+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23502.0 SingleEFlatPt2To100 2026D49PU+SingleElectronFlatPt2To100_GenSimHLBeamSpot+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23502.98 SingleEFlatPt2To100 2026D49PU_PMXS2+SingleElectronFlatPt2To100_GenSimHLBeamSpot+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23502.99 SingleEFlatPt2To100 2026D49PU_PMXS1S2+SingleElectronFlatPt2To100_GenSimHLBeamSpot+PREMIX_PremixHLBeamSpotPU+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23502.999 SingleEFlatPt2To100 2026D49PU_PMXS1S2PR+SingleElectronFlatPt2To100_GenSimHLBeamSpot+PREMIX_PremixHLBeamSpotPU+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
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
23201.0 SingleElectronPt10 2026D49+SingleElectronPt10_pythia8_GenSimHLBeamSpot+DigiTrigger+RecoGlobal+HARVESTGlobal
23202.0 SingleElectronPt35 2026D49+SingleElectronPt35_pythia8_GenSimHLBeamSpot+DigiTrigger+RecoGlobal+HARVESTGlobal
23203.0 SingleElectronPt1000 2026D49+SingleElectronPt1000_pythia8_GenSimHLBeamSpot+DigiTrigger+RecoGlobal+HARVESTGlobal
23291.0 SingleElectronPt15Eta1p7_2p7 2026D49+SingleElectronPt15Eta1p7_2p7_GenSimHLBeamSpot+DigiTrigger+RecoGlobal+HARVESTGlobal
23302.0 SingleEFlatPt2To100 2026D49+SingleElectronFlatPt2To100_GenSimHLBeamSpot+DigiTrigger+RecoGlobal+HARVESTGlobal
23401.0 SingleElectronPt10 2026D49PU+SingleElectronPt10_pythia8_GenSimHLBeamSpot+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23401.98 SingleElectronPt10 2026D49PU_PMXS2+SingleElectronPt10_pythia8_GenSimHLBeamSpot+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23401.99 SingleElectronPt10 2026D49PU_PMXS1S2+SingleElectronPt10_pythia8_GenSimHLBeamSpot+PREMIX_PremixHLBeamSpotPU+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23401.999 SingleElectronPt10 2026D49PU_PMXS1S2PR+SingleElectronPt10_pythia8_GenSimHLBeamSpot+PREMIX_PremixHLBeamSpotPU+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23402.0 SingleElectronPt35 2026D49PU+SingleElectronPt35_pythia8_GenSimHLBeamSpot+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23402.98 SingleElectronPt35 2026D49PU_PMXS2+SingleElectronPt35_pythia8_GenSimHLBeamSpot+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23402.99 SingleElectronPt35 2026D49PU_PMXS1S2+SingleElectronPt35_pythia8_GenSimHLBeamSpot+PREMIX_PremixHLBeamSpotPU+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23402.999 SingleElectronPt35 2026D49PU_PMXS1S2PR+SingleElectronPt35_pythia8_GenSimHLBeamSpot+PREMIX_PremixHLBeamSpotPU+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23403.0 SingleElectronPt1000 2026D49PU+SingleElectronPt1000_pythia8_GenSimHLBeamSpot+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23403.98 SingleElectronPt1000 2026D49PU_PMXS2+SingleElectronPt1000_pythia8_GenSimHLBeamSpot+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23403.99 SingleElectronPt1000 2026D49PU_PMXS1S2+SingleElectronPt1000_pythia8_GenSimHLBeamSpot+PREMIX_PremixHLBeamSpotPU+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23403.999 SingleElectronPt1000 2026D49PU_PMXS1S2PR+SingleElectronPt1000_pythia8_GenSimHLBeamSpot+PREMIX_PremixHLBeamSpotPU+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23491.0 SingleElectronPt15Eta1p7_2p7 2026D49PU+SingleElectronPt15Eta1p7_2p7_GenSimHLBeamSpot+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23491.98 SingleElectronPt15Eta1p7_2p7 2026D49PU_PMXS2+SingleElectronPt15Eta1p7_2p7_GenSimHLBeamSpot+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23491.99 SingleElectronPt15Eta1p7_2p7 2026D49PU_PMXS1S2+SingleElectronPt15Eta1p7_2p7_GenSimHLBeamSpot+PREMIX_PremixHLBeamSpotPU+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23491.999 SingleElectronPt15Eta1p7_2p7 2026D49PU_PMXS1S2PR+SingleElectronPt15Eta1p7_2p7_GenSimHLBeamSpot+PREMIX_PremixHLBeamSpotPU+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23502.0 SingleEFlatPt2To100 2026D49PU+SingleElectronFlatPt2To100_GenSimHLBeamSpot+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23502.98 SingleEFlatPt2To100 2026D49PU_PMXS2+SingleElectronFlatPt2To100_GenSimHLBeamSpot+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23502.99 SingleEFlatPt2To100 2026D49PU_PMXS1S2+SingleElectronFlatPt2To100_GenSimHLBeamSpot+PREMIX_PremixHLBeamSpotPU+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
23502.999 SingleEFlatPt2To100 2026D49PU_PMXS1S2PR+SingleElectronFlatPt2To100_GenSimHLBeamSpot+PREMIX_PremixHLBeamSpotPU+DigiTriggerPU+RecoGlobalPU+HARVESTGlobalPU
Found 25 compatible workflows inside relval_upgrade
matrix>
```

### Explore/Dump a specific workflow

Suppose now that you want to know what are the commands run by the workflow
`23291`. You could do by doing:

```
matrix> dumpWorkflowId 23291
23291.0 2026D49+SingleElectronPt15Eta1p7_2p7_GenSimHLBeamSpot+DigiTrigger+RecoGlobal+HARVESTGlobal
[1]: cmsDriver.py SingleElectronPt15Eta1p7_2p7_cfi  --conditions auto:phase2_realistic_T15 -n 10 --era Phase2C9 --eventcontent FEVTDEBUG --relval 9000,100 -s GEN,SIM --datatier GEN-SIM --beamspot HLLHC --geometry Extended2026D49

[2]: cmsDriver.py step2  --conditions auto:phase2_realistic_T15 -s DIGI:pdigi_valid,L1TrackTrigger,L1,DIGI2RAW,HLT:@fake2 --datatier GEN-SIM-DIGI-RAW -n 10 --geometry Extended2026D49 --era Phase2C9 --eventcontent FEVTDEBUGHLT

[3]: cmsDriver.py step3  --conditions auto:phase2_realistic_T15 -s RAW2DIGI,L1Reco,RECO,RECOSIM,PAT,VALIDATION:@phase2Validation+@miniAODValidation,DQM:@phase2+@miniAODDQM --datatier GEN-SIM-RECO,MINIAODSIM,DQMIO -n 10 --geometry Extended2026D49 --era Phase2C9 --eventcontent FEVTDEBUGHLT,MINIAODSIM,DQM

[4]: cmsDriver.py step4  --conditions auto:phase2_realistic_T15 -s HARVESTING:@phase2Validation+@phase2+@miniAODValidation+@miniAODDQM --scenario pp --filetype DQM --geometry Extended2026D49 --era Phase2C9 --mc  -n 100


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
2026
identity
machine
premix
nano

matrix> showWorkflow gpu
140.065506 Run3-2023_JetMET2022D_RecoPixelOnlyTripletsGPU
140.065512 Run3-2023_JetMET2022D_RecoECALOnlyGPU
140.065522 Run3-2023_JetMET2022D_RecoHCALOnlyGPU
11634.502 TTbar_14TeV+2021_Patatrack_PixelOnlyGPU
11634.503 TTbar_14TeV+2021_Patatrack_PixelOnlyGPU_Validation
11634.504 TTbar_14TeV+2021_Patatrack_PixelOnlyGPU_Profiling
11634.506 TTbar_14TeV+2021_Patatrack_PixelOnlyTripletsGPU
11634.507 TTbar_14TeV+2021_Patatrack_PixelOnlyTripletsGPU_Validation
11634.508 TTbar_14TeV+2021_Patatrack_PixelOnlyTripletsGPU_Profiling
11634.512 TTbar_14TeV+2021_Patatrack_ECALOnlyGPU
11634.513 TTbar_14TeV+2021_Patatrack_ECALOnlyGPU_Validation
11634.514 TTbar_14TeV+2021_Patatrack_ECALOnlyGPU_Profiling
11634.522 TTbar_14TeV+2021_Patatrack_HCALOnlyGPU
11634.523 TTbar_14TeV+2021_Patatrack_HCALOnlyGPU_Validation
11634.524 TTbar_14TeV+2021_Patatrack_HCALOnlyGPU_Profiling
11634.582 TTbar_14TeV+2021_Patatrack_AllGPU
11634.583 TTbar_14TeV+2021_Patatrack_AllGPU_Validation
11634.586 TTbar_14TeV+2021_Patatrack_AllTripletsGPU
11634.587 TTbar_14TeV+2021_Patatrack_AllTripletsGPU_Validation
11634.592 TTbar_14TeV+2021_Patatrack_FullRecoGPU
11634.593 TTbar_14TeV+2021_Patatrack_FullRecoGPU_Validation
11634.596 TTbar_14TeV+2021_Patatrack_FullRecoTripletsGPU
11634.597 TTbar_14TeV+2021_Patatrack_FullRecoTripletsGPU_Validation
11650.502 ZMM_14+2021_Patatrack_PixelOnlyGPU
11650.503 ZMM_14+2021_Patatrack_PixelOnlyGPU_Validation
11650.504 ZMM_14+2021_Patatrack_PixelOnlyGPU_Profiling
11650.506 ZMM_14+2021_Patatrack_PixelOnlyTripletsGPU
11650.507 ZMM_14+2021_Patatrack_PixelOnlyTripletsGPU_Validation
11650.508 ZMM_14+2021_Patatrack_PixelOnlyTripletsGPU_Profiling
12434.502 TTbar_14TeV+2023_Patatrack_PixelOnlyGPU
12434.503 TTbar_14TeV+2023_Patatrack_PixelOnlyGPU_Validation
12434.504 TTbar_14TeV+2023_Patatrack_PixelOnlyGPU_Profiling
12434.506 TTbar_14TeV+2023_Patatrack_PixelOnlyTripletsGPU
12434.507 TTbar_14TeV+2023_Patatrack_PixelOnlyTripletsGPU_Validation
12434.508 TTbar_14TeV+2023_Patatrack_PixelOnlyTripletsGPU_Profiling
12434.512 TTbar_14TeV+2023_Patatrack_ECALOnlyGPU
12434.513 TTbar_14TeV+2023_Patatrack_ECALOnlyGPU_Validation
12434.514 TTbar_14TeV+2023_Patatrack_ECALOnlyGPU_Profiling
12434.522 TTbar_14TeV+2023_Patatrack_HCALOnlyGPU
12434.523 TTbar_14TeV+2023_Patatrack_HCALOnlyGPU_Validation
12434.524 TTbar_14TeV+2023_Patatrack_HCALOnlyGPU_Profiling
12434.582 TTbar_14TeV+2023_Patatrack_AllGPU
12434.583 TTbar_14TeV+2023_Patatrack_AllGPU_Validation
12434.586 TTbar_14TeV+2023_Patatrack_AllTripletsGPU
12434.587 TTbar_14TeV+2023_Patatrack_AllTripletsGPU_Validation
12434.592 TTbar_14TeV+2023_Patatrack_FullRecoGPU
12434.593 TTbar_14TeV+2023_Patatrack_FullRecoGPU_Validation
12434.596 TTbar_14TeV+2023_Patatrack_FullRecoTripletsGPU
12434.597 TTbar_14TeV+2023_Patatrack_FullRecoTripletsGPU_Validation
12450.502 ZMM_14+2023_Patatrack_PixelOnlyGPU
12450.503 ZMM_14+2023_Patatrack_PixelOnlyGPU_Validation
12450.504 ZMM_14+2023_Patatrack_PixelOnlyGPU_Profiling
12450.506 ZMM_14+2023_Patatrack_PixelOnlyTripletsGPU
12450.507 ZMM_14+2023_Patatrack_PixelOnlyTripletsGPU_Validation
12450.508 ZMM_14+2023_Patatrack_PixelOnlyTripletsGPU_Profiling
gpu contains 55 workflows
matrix>
```

All commands come with dynamic TAB-completion. There's also a transient history
of the commands issued within a single session. Transient means that, after a
session is closed, the history is lost.
