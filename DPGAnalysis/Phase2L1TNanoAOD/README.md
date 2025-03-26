# Phase2-L1Nano
NanoAOD ntupler for Phase-2 L1 Objects

Initially an independent package here: https://github.com/cms-l1-dpg/Phase2-L1Nano/

## Setup

For more information on the latest L1T Phase 2 software developments in CMSSW see: https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideL1TPhase2Instructions#Development

Corresponding menu twiki section: https://twiki.cern.ch/twiki/bin/viewauth/CMS/PhaseIIL1TriggerMenuTools#Phase_2_L1_Trigger_objects_based


## Usage

### Via cmsDriver

One can append the L1Nano output to the `cmsDriver` command via the `NANO:@Phase2L1DPGwithGen` autoNANO handle, e.g.:
```bash
cmsDriver.py -s L1,L1TrackTrigger,L1P2GT,NANO:@Phase2L1DPGwithGen
```

Check `PhysicsTools/NanoAOD/python/autoNANO.py` for the way this command is defined.

Note that the step key `Phase2L1DPG` does not include the generator and reco-level objects used for MenuTools studies in the nano!
It is mostly created for workflow tests.

An example `cmsDriver` command for 14x files:
```bash
cmsDriver.py -s L1,L1TrackTrigger,L1P2GT,NANO:@Phase2L1DPGwithGen \
--conditions auto:phase2_realistic_T33 \
--geometry ExtendedRun4D110 \
--era Phase2C17I13M9 \
--eventcontent NANOAOD \
--datatier GEN-SIM-DIGI-RAW-MINIAOD \
--customise SLHCUpgradeSimulations/Configuration/aging.customise_aging_1000,Configuration/DataProcessing/Utils.addMonitoring,L1Trigger/Configuration/customisePhase2TTOn110.customisePhase2TTOn110 \
--filein /store/mc/Phase2Spring24DIGIRECOMiniAOD/TT_TuneCP5_14TeV-powheg-pythia8/GEN-SIM-DIGI-RAW-MINIAOD/PU200_AllTP_140X_mcRun4_realistic_v4-v1/2560000/11d1f6f0-5f03-421e-90c7-b5815197fc85.root \
--fileout file:output_Phase2_L1T.root \
--python_filename rerunL1_cfg.py \
--inputCommands="keep *, drop l1tPFJets_*_*_*, drop l1tTrackerMuons_l1tTkMuonsGmt*_*_HLT" \
--mc \
-n 10 --nThreads 4 --no_exec
```

### Workflows

Two upgrade workflows are implemented to test/run this nano (`Phase2L1DPG`) after the full DigiTrigger chain after the complete L1:
* `.781` - produces NANO in addition to FEVTDEBUG
* `.782` - produces only NANO

And can be executed as e.g. `runTheMatrix.py --what upgrade -i all --ibeos -l 29634.782`. See [here the readme of `runTheMatrix`](https://github.com/cms-sw/cmssw/tree/master/Configuration/PyReleaseValidation/scripts#interactive-runthematrix-shell) and [here a list of workflows](https://github.com/cms-sw/cmssw/tree/master/Configuration/PyReleaseValidation).

Note that if tou want to include the Gen/Offline references, you need to change the import function in the config after the

## Output

The output file is a nanoAOD file with the output branches in the `Events` tree.

An overview of the corresponding content is shown here: https://alobanov.web.cern.ch/L1T/Phase2/L1Nano/l1menu_nano_V38_1400pre3V9_doc_report.html

Size report: https://alobanov.web.cern.ch/L1T/Phase2/L1Nano/l1menu_nano_V38_1400pre3V9_size_report.html

Example:

```python
'run',
'luminosityBlock',
'event',
'L1tkPhoton_saId',
'L1tkPhoton_hwEta',
'L1tkPhoton_hwIso',
'L1tkPhoton_hwPhi',
```

This can be easily handled with [`uproot/awkward`](https://gitlab.cern.ch/cms-podas23/dpg/trigger-exercise/-/blob/solutions/1_Intro_NanoAwk_Analysis_Solution.ipynb) like this:

```python
f = uproot.open("l1nano.root")
events = f["Events"].arrays() 
```

### P2GT emulator decisions
The GT emulator decisions are stored like this now:
```
L1_pSingleTkMuon22_final # seed name and final for post prescale value
```
