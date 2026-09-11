# Running the DQM/Validation code
## Phase2 Tracker DQM:

Producing DQM plots is split into two parts. 

Input file is GEN-SIM-RECO or GEN-SIM-DIGI-RAW .root file, inside ```test/dqmstep_phase2tk_cfg.py```.

Step 1 of the DQM plotting: 
```
cmsRun dqmstep_phase2tk_cfg.py
```

The output file ```step3_pre4_inDQM.root``` is then used as input in the step 2 (harvesting):
```
cmsRun harvestingstep_phase2tk_cfg.py
```

The final output is ```DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root``` with DQM histograms.

## Phase2 C-RACK DQM:

C-RACK is Cosmic Rack test stand in TIF with up to 6 Ladders of 12 2S modules.

To produce DQM plots on C-RACK for MC and data there are dedicated scripts for both DQM and Harvesting steps in the /test/ folder.

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

DQM plots to be produced only for C-RACK could be set with ```switch = false```, and enabled inside the plugin .cff, like ```python/Phase2OTMonitorCluster_cff.py```. 
Addtionally, for real data, the product source may need to be changed. ```clusterSrc = cms.InputTag("Unpacker", "", "UNPACK") # only for unpacked clusters from CRACK, not MC```
# Adding DQM/Validation Plots
## Overview
The DQM and validation code are structurally similar, so the steps to add a histogram to each are roughly the same. All the booking/filling is handled in the plugins. Digi histograms and "debug" histograms have a few extra steps. 

Most histograms use booking & filling "depth" to determine where the histogram will appear. A histogram can be created in `InnerTracker/Barrel`, `InnerTracker/Barrel/mI`, `InnerTracker/Barrel/mO`, etc. without having to manually type the foldername.

| Depth | Inner Tracker Structure       | Outer Tracker Structure       |
| ----- | ----------------------------- | ----------------------------- |
| 1     | InnerTracker                  | OuterTracker                  |
| 2     | Barrel, forward, or endcap    | Barrel or endcap              |
| 3     | Shell/Half-cylinder           | (Endcaps only) Side           |
| 4     | (Endcaps only) Ring           | (Endcaps only) Ring           |
| 5     | (Endcaps only) Wheel          | (Endcaps only) Wheel          |
| 6     | Barrel layer OR ring in wheel | Barrel layer OR ring in wheel |

The depth is used to get the foldername & "pretty" information in `phase2tkutil::getHistoId(detid, TrackerTopology, phi, DEPTH, pretty)`.

Example getHistoId output with depth 4:

> /Endcaps/EndcapPix/mI/Ring1

Example getHistoId pretty output, depth 4:

> IT endcap EPix shell mI Ring1

## 1) Add Parameters

ParameterSets for each plugin are all defined in ```fillDescriptions``` (except for DQM digis). `phase2tkutil` is used to add ParameterSets. 

* desc: the ParameterSetDescription
* "psetKey": Name of the PSet
* "histName": used to create the filepath
* "histTitle": displayed in the histogram itself. Use {} to add pretty position information for depth-wise histograms
* "xlabel": x axis label
* "ylabel": y axis label
* "nbins": Number of x bins
* "xmin": lower bound of x axis
* "xmax": upper bound of x axis

```cpp
  phase2tkutil::add1DDesc(desc,
                          "NClustersLayer",
                          "Num_Clusters_Per_Event",
                          "Number Of Clusters per event in {}",
                          "Number of Clusters per event",
                          "Number of events",
                          150,
                          0.0,
                          250000.0);
  ```
  
## 2) Book Histogram
There are two histogram booking methods that should be used depending on whether this histogram should be booked in the "detector structure"/depth-wise folders or in another folder. Each histogram needs to be declared as a MonitorElement at the top of the plugin. Histograms that are going to be booked using depth need to be in the ME struct, while histograms in a single specific folder (i.e. global XY histogram to go in /Positions) should be declared seperately.

```cpp
  struct ClusterMEs { // depth-wise histos go in here
    MonitorElement* nClusters = nullptr;
    MonitorElement* ClusterSize = nullptr;
    MonitorElement* ClusterSizeX = nullptr;
    MonitorElement* ClusterSizeY = nullptr;
    MonitorElement* ClusterCharge = nullptr;
    unsigned int clusterCounter{0};
    MonitorElement* ClusterPos_Mod_Ladder = nullptr;            // Only in Barrel and LAYER
    MonitorElement* XY_byWheel = nullptr;                       // Only in ENDCAP_WHEEL
    std::vector<MonitorElement*> ClusterPos_BLayer_Mod_Ladder;  // Only in Barrel and SUBSTRUCTURE
  };
  // Position histos
  MonitorElement* globalXY_barrel_;
  MonitorElement* globalXY_endcap_;
  MonitorElement* globalRZ_barrel_;
  MonitorElement* globalRZ_endcap_;
  ```

```bookHistograms``` is used to book a histogram in a specific folder. Use the ```ibooker.cd()``` method to select the folder and book the histogram.

```cpp !
ibooker.cd(top_folder + "myFolder/")
myHisto = phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("newHisto_PSet"), ibooker)
```

```bookLayerHistos``` books histograms in every depth by default. By using if statements, you can choose specific depths or module types. Most of the logic is already there, so book your histogram in the appropriate place using phase2tkutil.

```cpp
// The booking process iterates over every depth, from top-level (IT) down to layer 
for (enum Level bookingDepth = IT; bookingDepth <= LAYER; bookingDepth = Level(bookingDepth + 1)) {
    // Skip booking for barrel det_ids in endcap-only depths
    if ((bookingDepth == ENDCAP_RING || bookingDepth == ENDCAP_WHEEL) &&
        DetId(det_id).subdetId() == PixelSubdetector::PixelBarrel)
      continue;

    std::string folderName = phase2tkutil::getHistoId(det_id, tTopo_, detPos.phi(), bookingDepth, false);
    std::string prettyName = phase2tkutil::getHistoId(det_id, tTopo_, detPos.phi(), bookingDepth, true);

    std::map<std::string, ClusterMEs>::iterator pos = layerMEs_.find(folderName);
    // Prevent duplicate MEs
    if (pos == layerMEs_.end()) {
      ibooker.cd();
      ibooker.setCurrentFolder(subdir + "/" + folderName);
      // now book your histograms
```

```cpp !
// to book at all depths:
local_mes.myHisto = phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("newHisto_PSet"), ibooker, prettyName);
// to book at a specific depth:
if (bookingDepth == ENDCAP_WHEEL) 
        local_mes.myWheelHistogram = phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("newWheelHisto_PSet"), ibooker, prettyName);
```
## 3) Fill Histogram
```analyse``` is called after every event to fill histograms. **If you have a histogram that does not use depth, you will have to use the detId to check if your histogram needs to be filled**. This example is from the OTCluster code & fills the histogram if the detId is part of the barrel. This histogram counts the number of clusters in each barrel layer.

```cpp
if (detId.subdetId() == SiStripSubdetector::TOB)
    if (myHistogram_Barrel_)
        myHistogram_Barrel_->Fill(tTopo_->layer(detId));
```
However, if you have used depth, there should not be a need to check which substructure you are in. **Be aware you will still need to check module type if your histogram is for a specific type!** 
```cpp
for (enum Level fillingDepth = IT; fillingDepth <= LAYER; fillingDepth = Level(fillingDepth + 1)) {
        // Skip filling for barrel detIds on endcap-only depths
        if ((fillingDepth == ENDCAP_RING || fillingDepth == ENDCAP_WHEEL) &&
            DetId(detId).subdetId() == PixelSubdetector::PixelBarrel)
          continue;
        std::string folderkey = phase2tkutil::getHistoId(detId, tTopo_, detPos.phi(), fillingDepth, false);
        auto local_mesIT = layerMEs_.find(folderkey);
        if (local_mesIT == layerMEs_.end())
          continue;
        ClusterMEs& local_mes = local_mesIT->second;

        // This will only exist in endcap wheels, so no need to check!
        if (local_mes.XY_byWheel)
          local_mes.XY_byWheel->Fill(gx, gy);
        // Ditto for barrel layers
        if (local_mes.ClusterPos_Mod_Ladder)
          local_mes.ClusterPos_Mod_Ladder->Fill(signedModule, signedLadder);
```

## 4) Disable & Enable Histograms
Histograms are enabled by default. If you have a debug or geometry-dependent histogram (CRACK), it will have to be disabled in the plugin's cff file. Then, you can create a second configuration in that file that will have the histogram enabled. You can also change the parameter set of histograms that are usually enabled (for example, in the Cosmic Rack you do not need ranges that accomodate the full tracker size, etc.)

```python
import FWCore.ParameterSet.Config as cms

from DQM.SiTrackerPhase2.Phase2OTMonitorCluster_cfi import Phase2OTMonitorCluster 

clusterMonitorOT = Phase2OTMonitorCluster.clone(
        PositionOfClusters_2S = Phase2OTMonitorCluster.PositionOfClusters_2S.clone(
            switch = cms.bool(False)
        ),
        CrackOverview = Phase2OTMonitorCluster.CrackOverview.clone(
            switch = cms.bool(False)
        )
)
clusterMonitorCRACK = Phase2OTMonitorCluster.clone(
    # Histograms that are usually set to switch = False in full tracker
    PositionOfClusters_2S = Phase2OTMonitorCluster.PositionOfClusters_2S.clone(
        ...
```
You should then use this special configuration in your config file to run.

## Digis note
Digis do not have the parameters defined in the plugin. The default config is in the python cff, which is then cloned and edited for OT/IT digis. Add your histogram parameters in there.
