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

To produce DQM plots on C-RACK (for MC, tested, and eventually data, yet to be tested) there are dedicated scripts for both DQM and Harvesting steps in the /test/ folder.

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
# Adding DQM/Validation Plots

The DQM and validation code are structurally similar, so the steps to add a histogram to each are roughly the same. All the booking/filling is handled in the plugins. Digi histograms and "debug" histograms have a few extra steps. 

## 1) Add Parameters

ParameterSets for each plugin are all defined in ```fillDescriptions``` except for DQM digis. 

* name: used to create the filepath
* title: displayed in the histogram itself. Takes the form of "Title;x axis label;y axis label" (but the axis labels can be left blank). 
* switch: used to "switch on" the booking/filling of the histogram. Most are on by default, but if you only want the histogram to be used with a certain geometry or for debugging, please set to false.

```c++
 {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Cluster_Size_S");
    psd0.add<std::string>("title", "Cluster_Size_S;Cluster size (strip sensor);");
    psd0.add<double>("xmin", -0.5);
    psd0.add<double>("xmax", 30.5);
    psd0.add<int>("NxBins", 31);
    psd0.add<bool>("switch", true);
    desc.add<edm::ParameterSetDescription>("newHisto_PSet", psd0); //PSet name
  }
  ```
  
## 2) Book Histogram
There are two histogram booking methods that should be used depending on whether this histogram should be booked per layer/ring/wheel or in a specific folder. Each histogram needs to be declared as a MonitorElement at the top of the plugin. Histograms that are going to be booked per layer need to be in the ME struct, while histograms in a single specific folder (i.e. a barrel overview) should be declared seperately.

```c++
private:
  struct ClusterMEs { // layer wise histograms go here.
    MonitorElement* nClusters_P = nullptr;
    MonitorElement* ClusterSize_P = nullptr;

    MonitorElement* nClusters_S = nullptr;
    MonitorElement* ClusterSize_S = nullptr;
  };
  // regular histograms should be declared down here.
  MonitorElement* numberClusters_;
  MonitorElement* numberClusters_Barrel_;
  MonitorElement* globalXY_P_;
  MonitorElement* globalRZ_P_;
  MonitorElement* globalXY_S_;
  MonitorElement* globalRZ_S_;
  ```

```bookHistograms``` is used to book a histogram in a specific folder. Use the ```ibooker.cd()``` method to select the folder ```ibooker.cd(top_folder + "yourFolder/")``` and book the histogram.

```c++!
myHisto = phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("newHisto_PSet"), ibooker)
```

```bookLayerHistos``` books histograms in every layer, ring, and wheel. If your histogram has some special requirements (like showing only PSP data) you should use if statements to ensure it only books in relevant layers. Most of the logic is already there,, so book your histogram in the appropriate place using phase2tkutil.

```c++!
local_mes.myHisto = phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("newHisto_PSet"), ibooker)
```
## 3) Fill Histogram
```analyse``` is called after every event to fill histograms. Use the detId to check if your histogram needs to be filled, and then fill it. This example is from the OTCluster code & fills the histogram if the detId is part of the barrel. This histogram counts the number of clusters in each barrel layer.

```c++
if (detId.subdetId() == SiStripSubdetector::TOB) {
    myHistogram_Barrel_->Fill(tTopo_->layer(detId));
}
```

## 4) Special case: Enable Histogram
If you have a histogram that should only be enabled in a special case, you should either use a python cff file to enable it manually, or edit the test script to enable the histogram. You can also change the parameter set of histograms that are usually enabled (for example, if you have a funky geometry like in the Cosmic Rack you do not need ranges that accomodate the full tracker size, etc.)

```python
import FWCore.ParameterSet.Config as cms

from DQM.SiTrackerPhase2.Phase2OTMonitorCluster_cfi import Phase2OTMonitorCluster as _Phase2OTMonitorCluster

myClusterMonitor = _Phase2OTMonitorCluster.clone(
    # Histograms that are usually set to switch = False
    myDebugHistogram = _Phase2OTMonitorCluster.myDebugHistogram.clone(
        switch = cms.bool(True)
    )
    # Histograms where you want to change the ranges
    myChangingHistogram = _Phase2OTMonitorCluster.myChangingHistogram.clone(
        NxBins = cms.int32(1016),
        xmin = cms.double(0.5),
        xmax = cms.double(1016.5),
    )
)
```
You should then use this special cff in your config file to run.

## Digis note
Digis do not have the parameters defined in the plugin. The default config is in the python cff, which is then cloned and edited for OT/IT digis. Add your histogram parameters in there.
