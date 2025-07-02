# CRACK for PhaseII:

## Geometry generation

This section provides complete instructions for creating the CRACK geometry.
If you only wish to use the geometry without generating it, please skip to the next section.

This setup does not use TKLayout. It is a manual setup to create a custom standalone Tracker, in this case, the CRACK geometry.
If you want to run a standard GEN-SIM workflow instead, please refer to the official instructions:
[here](https://github.com/cms-sw/cmssw/tree/master/Configuration/Geometry) and [here](https://github.com/cms-sw/cmssw/tree/master/Configuration/PyReleaseValidation).

The CRACK geometry is defined through a set of XML files registered under the name TCRACK in dictCRACKIIGeometry.py.

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

## Gen-Sim step 

To run the Gen-Sim step for the CRACK, run the following command: 
```
cmsRun myTrackerOnly_cfg.py
```



## Run the workflow  (NEEDS TO BE UPDATED)
500 is the new detector version for the standelone CRACK. The following step creates available workflows for D500 sush as a simple the GEN-SIM step

The new Geometry is integrated in the full matrix, and several new workflow made available. a workflow is a set of GEN-SIM-RECO etc steps with different configurations. For further details on this step please refer to this presentation [here](https://indico.cern.ch/event/1296370/contributions/5449497/attachments/2664526/4616810/TkGeom_handover_AdeWit.pdf)

To check the available workflows for D500 

```
 runTheMatrix.py --what upgrade --showMatrix | grep D500 > log
```

To run a simple GEN-SIM step on the standelone version using cosmics you can use the workflow 32854.0 

```
runTheMatrix.py -l 32854.0 -t 4 --what upgrade --command="-n 1"

```
This will simply run a cmsDriver command for the GEN-SIM step with cosmic muons. 


