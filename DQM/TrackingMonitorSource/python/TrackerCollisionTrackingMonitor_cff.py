import FWCore.ParameterSet.Config as cms

from DQM.TrackingMonitorSource.pset4GenericTriggerEventFlag_cfi import *

import DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi

# Clone for TrackingMonitor for all PDs but MinBias ###
TrackerCollisionTrackMonCommon = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone()
TrackerCollisionTrackMonCommon.genericTriggerEventPSet = genericTriggerEventFlag4fullTracker
TrackerCollisionTrackMonCommon.setLabel("TrackerCollisionTrackMonCommon")

# Clone for TrackingMonitor for MinBias ###
TrackerCollisionTrackMonMB = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone()
TrackerCollisionTrackMonMB.genericTriggerEventPSet = genericTriggerEventFlag4fullTrackerAndHLTdb
TrackerCollisionTrackMonMB.doPrimaryVertexPlots = cms.bool(True)
TrackerCollisionTrackMonMB.setLabel("TrackerCollisionTrackMonMB")

