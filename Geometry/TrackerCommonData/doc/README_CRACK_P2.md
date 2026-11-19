# CRACK for Phase-II:

This was last validated in CMSSW_20_1_0_pre1.

## Geometry generation

CHECK: WHY NOT MENTIONS Geometry/TrackerCommonData/data/CRack_PhaseII/ ???

This section provides complete instructions for creating the CRACK geometry.
If you only wish to use the geometry without generating it, please skip to the next section.

This setup does not use TKLayout. It is a manual setup to create a custom standalone Tracker, in this case, the CRACK geometry.
If you want to run a standard GEN-SIM workflow instead, please refer to the official instructions:
[here](https://github.com/cms-sw/cmssw/tree/master/Configuration/Geometry) and [here](https://github.com/cms-sw/cmssw/tree/master/Configuration/PyReleaseValidation).

+ The CRACK geometry is defined through a set of XML files registered under the name `TCRACK` in `dictCRACKIIGeometry.py`.
+ If you wish to modify the current CRACK geometry, you can find instructions in [DPG-presentation1](https://indico.cern.ch/event/1567987/contributions/6670173/attachments/3125573/5543509/CRACK_DPG_29082025.pdf) and [DPG-presentation2](https://indico.cern.ch/event/1628558/contributions/6877385/attachments/3202037/5700480/Phase2DPG-Meeting_16012026.pdf) 

To create a new detector version with the standalone CRACK geometry, run the following commands:

```
git cms-addpkg Geometry/CMSCommonData
git cms-addpkg Configuration/Geometry
scram b -j 8
cd Configuration/Geometry
python3 ./scripts/generateCRACKIIGeometry.py -D 500
```
## Geometry validation:

To validate the geometry, two options are available:

1. Fireworks Geometry Display (Linux only)

2. ROOT Macro Visualization (Linux and macOS)

### Fireworks (Linux only)

```
git cms-addpkg Fireworks
cmsRun Fireworks/Geometry/python/dumpSimGeometry_cfg.py tag=Run4 version=D500
LD_PRELOAD="/lib64/libLLVM-17.so"
cmsShow --sim-geom-file cmsSimGeom-Run4D500.root -c Fireworks/Core/macros/simGeo.fwc
```

### ROOT (both Linux and macOS)
Alternatively, use a ROOT macro to visualize the geometry. This method works on both Linux and macOS. On macOS, run the macro locally (not via ssh to lxplus) to avoid X11 forwarding issues.

```
 root Geometry_plotter.C
```
# Create P2 C-RACK Monte Carlo

This simulates cosmic rays traversing the CRACK. The CRACK (D500) geometry is integrated in CMSSW and can be called and used as any other CMS detector geometry.

## Gen-Sim step

```
cmsDriver.py UndergroundCosmicSPLooseMu_cfi -s GEN,SIM -n 10000 --conditions auto:phase2_realistic_0T --beamspot DBrealisticHLLHC --datatier GEN-SIM --eventcontent FEVTDEBUG --geometry ExtendedRun4D500 --era phase2_tracker --fileout file:step1.root --nThreads 16 --python step1.py --customise SimTracker/Configuration/customise_P2_CRACK.step1
```
Here, the --customise option refers to a python file that disables simulation of detectors other than the Tracker, and configures the cosmic ray generator appropriately for the CRACK.

## DIGI step:

This does Tracker digitisation and adds offline clusters, TTClusters & TTStubs, plus truth & truth-association info.

```
cmsDriver.py -s DIGI:pdigi_valid,L1TrackTrigger --conditions auto:phase2_realistic_0T --datatier GEN-SIM-DIGI --eventcontent FEVTDEBUG --geometry ExtendedRun4D500 --era phase2_tracker --magField 0T -n -1 --filein file:step1.root --fileout file:step2.root --nThreads 16  --python step2.py --customise SimTracker/Configuration/customise_P2_CRACK.step2

```

## Packer and Unpacker: 

Instructions for running the packer/unpacker can be found in the CRACK-unpacker repository [CRACK-unpacker](https://github.com/P2-Tracker-BES-SW/cmssw/tree/unpackers_20_1_0_pre3/EventFilter/Phase2TrackerRawToDigi/doc). you have then to switch to the specific CRACK branch `rebase_unpackers_to_16_0_0_crack`. If you cannot find this branch, its name may have changed; in that case, look for another branch containing `CRACK` in its name.


## DQM 

DQM instructions can be found in this repository [CRACK-DQM](https://github.com/cms-sw/cmssw/blob/master/DQM/SiTrackerPhase2/test/README.md#phase2-c-rack-dqm)


