import FWCore.ParameterSet.Config as cms

from DQM.TrackingMonitorSource.pset4GenericTriggerEventFlag_cfi import *
from DQM.TrackingMonitor.LogMessageMonitor_cfi import *

# Clone for all PDs but MinBias ####
LogMessageMonCommon = LogMessageMon.clone(
    genericTriggerEventPSet = genericTriggerEventFlag4fullTracker
)

# Clone for MinBias ###
LogMessageMonMB = LogMessageMon.clone(
    genericTriggerEventPSet = genericTriggerEventFlag4fullTrackerAndHLTdb
)
