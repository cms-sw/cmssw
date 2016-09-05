import FWCore.ParameterSet.Config as cms

from DQM.TrackingMonitorSource.pset4GenericTriggerEventFlag_cfi import *

# Clone for TrackingMonitor for all PDs but MinBias ###
import DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi
TrackerCollisionTrackMonCommon = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone()
TrackerCollisionTrackMonCommon.genericTriggerEventPSet = genericTriggerEventFlag4fullTracker
TrackerCollisionTrackMonCommon.setLabel("TrackerCollisionTrackMonCommon")

# Clone for TrackingMonitor for MinBias ###
import DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi
TrackerCollisionTrackMonMB = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone()
TrackerCollisionTrackMonMB.genericTriggerEventPSet = genericTriggerEventFlag4fullTrackerAndHLTdb
TrackerCollisionTrackMonMB.doPrimaryVertexPlots    = cms.bool(True)
TrackerCollisionTrackMonMB.setLabel("TrackerCollisionTrackMonMB")

