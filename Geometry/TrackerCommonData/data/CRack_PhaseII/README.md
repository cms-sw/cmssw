# CRACK for Phase-II:

To use the most up-to-date CRACK software, you should use `CMSSW_17_0_0_pre2`.
```
cmsrel CMSSW_17_0_0_pre2
cd CMSSW_17_0_0_pre2/src
cmsenv

```
## Geometry generation

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
# Run the CRACK workflow 
## Gen-Sim step 

The CRACK (D500) geometry is integrated in CMSSW and can be called and used as any other CMS detector geometry in the header of your python configuration file.  

To run the cosmic-ray generation and simulation for the CRACK geometry, execute:
```
cmsRun step1_cosmics_for_crack.py
```

This creates a `step1.root` file. To analyze the hits in the ROOT file, you can use the [SimHitAnalyzer](https://github.com/hayfasfar/SimHitAnalyzer/tree/master/SimHitAnalyzer).

## DIGI step: digitization, clusters, stubs and cluster1D objects 

To perform the digitization, clustering, stub formation, and 1D cluster object creation, run:
```

cmsRun step2_digi_ttclusters_ttstubs_cluster1Dobj.py

```
## Packer and Unpacker: 

Instructions for running the packer/unpacker can be found in the CRACK-unpacker repository [CRACK-unpacker](https://github.com/P2-Tracker-BES-SW/cmssw/tree/unpackers_16_0_0_pre1/EventFilter/Phase2TrackerRawToDigi/doc). you have then to switch to the specific CRACK branch `rebase_unpackers_to_16_0_0_crack`. If you cannot find this branch, its name may have changed; in that case, look for another branch containing `CRACK` in its name.


## DQM 

DQM instructions can be found in this repository [CRACK-DQM](https://github.com/cms-sw/cmssw/blob/master/DQM/SiTrackerPhase2/test/README.md#phase2-c-rack-dqm)


