import FWCore.ParameterSet.Config as cms

from DQM.TrackingMonitorSource.pset4GenericTriggerEventFlag_cfi import *

# Clone for TrackingMonitor for all PDs but MinBias ###
from DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi import *
TrackerCollisionTrackMonCommon = TrackerCollisionTrackMon.clone(
    genericTriggerEventPSet = genericTriggerEventFlag4fullTracker
)
TrackerCollisionTrackMonCommon.setLabel("TrackerCollisionTrackMonCommon")

# Clone for TrackingMonitor for ZeroBias ###
from DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi import *
TrackerCollisionTrackMonMB = TrackerCollisionTrackMon.clone(
    genericTriggerEventPSet = genericTriggerEventFlag4fullTrackerAndHLTdb,
    doPrimaryVertexPlots = True
)
TrackerCollisionTrackMonMB.setLabel("TrackerCollisionTrackMonMB")

# Clone for TrackingMonitor for ZeroBias, no hip, no OOT pu (1st collision after abort gap) ###
TrackerCollisionTrackMonZBnoHIPnoOOT = TrackerCollisionTrackMon.clone(
    genericTriggerEventPSet = genericTriggerEventFlag4fullTrackerAndHLTnoHIPnoOOTdb,
    doPrimaryVertexPlots = True
)
TrackerCollisionTrackMonZBnoHIPnoOOT.setLabel("TrackerCollisionTrackMonZBnoHIPnoOOT")

# Clone for TrackingMonitor for ZeroBias, hip, no OOT pu (1st collision in train) ###
TrackerCollisionTrackMonZBHIPnoOOT = TrackerCollisionTrackMon.clone(
    genericTriggerEventPSet = genericTriggerEventFlag4fullTrackerAndHLTHIPnoOOTdb,
    doPrimaryVertexPlots = True
)
TrackerCollisionTrackMonZBHIPnoOOT.setLabel("TrackerCollisionTrackMonZBHIPnoOOT")

# Clone for TrackingMonitor for ZeroBias, hip, OOT pu (1st collision after train) ###
TrackerCollisionTrackMonZBHIPOOT = TrackerCollisionTrackMon.clone(
    genericTriggerEventPSet = genericTriggerEventFlag4fullTrackerAndHLTHIPOOTdb,
    doPrimaryVertexPlots = True
)
TrackerCollisionTrackMonZBHIPOOT.setLabel("TrackerCollisionTrackMonZBHIPOOT")

