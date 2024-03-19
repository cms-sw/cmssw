To run the L1 tracking & create a TTree of tracking performance:

cmsRun L1TrackNtupleMaker_cfg.py

By setting variable L1TRKALGO inside this script, you can change which 
L1 tracking algo is used. It defaults to HYBRID. 

For the baseline HYBRID algo, which runs Tracklet pattern reco followed
by KF track fit, TrackFindingTracklet/interface/Settings.h configures the pattern reco stage, (although some parameters there are overridden by l1tTTTracksFromTrackletEmulation_cfi.py).
The KF fit is configured by the constructor of TrackFindingTMTT/src/Settings.cc.

The ROOT macros L1TrackNtuplePlot.C & L1TrackQualityPlot.C make track 
performance & BDT track quality performance plots from the TTree. 
Both can be run via makeHists.csh .

The optional "NewKF" track fit can be run by changing L1TRKALGO=HYBRID_NEWKF. It corresponds to the curent FW, but is is not yet the default, as only a basic duplicate track removal is available for it. It is configured via 
TrackTrigger/python/ProducerSetup_cfi.py, (which also configures the DTC).

For experts
============

1) To make plots to monitor data rates assicuated to truncation after each step in the tracklet pattern reco algo, set writeMonitorData_ = true in Settings.h . This creates txt files, which the ROOT macros in https://github.com/cms-L1TK/TrackPerf/tree/master/PatternReco can then use to study truncation of individual algo steps within tracklet chain.
