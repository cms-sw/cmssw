To run L1 tracking & create TTree of tracking performance:

cmsRun L1TrackNtupleMaker_cfg.py

By setting variable L1TRKALGO inside this script, you can change the 
L1 tracking algo used. 

For the baseline HYBRID algo, which runs Tracklet pattern reco followed
by KF track fit, TrackFindingTracklet/interface/Settings.h configures the pattern reco, (although some 
parameters there are overridden by l1tTTTracksFromTrackletEmulation_cfi.py).
The KF fit is configued by the constructor of TrackFindingTMTT/src/Settings.cc.

The ROOT macros L1TrackNtuplePlot.C & L1TrackQualityPlot.C make tracking 
performance & BDT track quality performance plots from the TTree. 
Both can be run via makeHists.csh .

The optional "NewKF" track fit, (which is not yet baseline, as no duplicate
track removal is compatible with it), is configured via 
TrackTrigger/python/ProducerSetup_cfi.py, (which also configures the DTC).
