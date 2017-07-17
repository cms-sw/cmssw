import FWCore.ParameterSet.Config as cms

from DQM.TrackingMonitorSource.pset4GenericTriggerEventFlag_cfi import *

# Clone for TrackingMonitor for all PDs but MinBias ###
import DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi
TrackerCollisionTrackMonCommon = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone()
TrackerCollisionTrackMonCommon.genericTriggerEventPSet = genericTriggerEventFlag4fullTracker
TrackerCollisionTrackMonCommon.setLabel("TrackerCollisionTrackMonCommon")

# Clone for TrackingMonitor for ZeroBias ###
import DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi
TrackerCollisionTrackMonMB = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone()
TrackerCollisionTrackMonMB.genericTriggerEventPSet = genericTriggerEventFlag4fullTrackerAndHLTdb
TrackerCollisionTrackMonMB.doPrimaryVertexPlots    = cms.bool(True)
TrackerCollisionTrackMonMB.setLabel("TrackerCollisionTrackMonMB")

# Clone for TrackingMonitor for ZeroBias, no hip, no OOT pu (1st collision after abort gap) ###
TrackerCollisionTrackMonZBnoHIPnoOOT = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone()
TrackerCollisionTrackMonZBnoHIPnoOOT.genericTriggerEventPSet = genericTriggerEventFlag4fullTrackerAndHLTnoHIPnoOOTdb
TrackerCollisionTrackMonZBnoHIPnoOOT.doPrimaryVertexPlots    = cms.bool(True)
TrackerCollisionTrackMonZBnoHIPnoOOT.setLabel("TrackerCollisionTrackMonZBnoHIPnoOOT")

# Clone for TrackingMonitor for ZeroBias, hip, no OOT pu (1st collision in train) ###
TrackerCollisionTrackMonZBHIPnoOOT = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone()
TrackerCollisionTrackMonZBHIPnoOOT.genericTriggerEventPSet = genericTriggerEventFlag4fullTrackerAndHLTHIPnoOOTdb
TrackerCollisionTrackMonZBHIPnoOOT.doPrimaryVertexPlots    = cms.bool(True)
TrackerCollisionTrackMonZBHIPnoOOT.setLabel("TrackerCollisionTrackMonZBHIPnoOOT")

# Clone for TrackingMonitor for ZeroBias, hip, OOT pu (1st collision after train) ###
TrackerCollisionTrackMonZBHIPOOT = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone()
TrackerCollisionTrackMonZBHIPOOT.genericTriggerEventPSet = genericTriggerEventFlag4fullTrackerAndHLTHIPOOTdb
TrackerCollisionTrackMonZBHIPOOT.doPrimaryVertexPlots    = cms.bool(True)
TrackerCollisionTrackMonZBHIPOOT.setLabel("TrackerCollisionTrackMonZBHIPOOT")

