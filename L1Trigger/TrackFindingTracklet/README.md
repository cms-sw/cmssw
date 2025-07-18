To run the L1 tracking & create a TTree of tracking performance:  

cmsRun L1TrackNtupleMaker_cfg.py

By setting variable L1TRKALGO inside this script, you can change which L1 tracking algo is used. It defaults to HYBRID, which runs Tracklet pattern reco followed by old Kalman track fit.

The version of the hybrid algorithm that corresponds to the current firmware, and includes the new Kalman track fit, can be run by changing L1TRKALGO=HYBRID_NEWKF. It is not yet the default for MC production, as it's tracking performance is not quite has good as HYBRID. e.g. Only a basic duplicate track removal is available for it.

Displaced Hybrid tracking can be run by setting L1TRKALGO=HYBRID_DISPLACED.

The ROOT macros L1TrackNtuplePlot.C & L1TrackQualityPlot.C make track performance & BDT track quality performance plots from the TTree. Both can be run via makeHists.csh .

If you need to modify the cfg params of the algorithm, then TrackFindingTracklet/interface/Settings.h configures the pattern reco stage, (although some parameters there are overridden by l1tTTTracksFromTrackletEmulation_cfi.py). The old KF fit is configured by the constructor of TrackFindingTMTT/src/Settings.cc. The DTC and new KF fit are configured via TrackTrigger/python/ProducerSetup_cfi.py.


For experts
============

1) To make plots to monitor data rates assicuated to truncation after each step in the tracklet pattern reco algo, set writeMonitorData_ = true in Settings.h . This creates txt files, which the ROOT macros in https://github.com/cms-L1TK/TrackPerf/tree/master/PatternReco can then use to study truncation of individual algo steps within tracklet chain.
