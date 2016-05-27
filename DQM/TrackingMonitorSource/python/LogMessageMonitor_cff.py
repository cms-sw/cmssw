import FWCore.ParameterSet.Config as cms

from DQM.TrackingMonitorSource.pset4GenericTriggerEventFlag_cfi import *
import DQM.TrackingMonitor.LogMessageMonitor_cfi

# Clone for all PDs but MinBias ####
LogMessageMonCommon = DQM.TrackingMonitor.LogMessageMonitor_cfi.LogMessageMon.clone()
LogMessageMonCommon.genericTriggerEventPSet = genericTriggerEventFlag4fullTracker

# Clone for MinBias ###
LogMessageMonMB = DQM.TrackingMonitor.LogMessageMonitor_cfi.LogMessageMon.clone()
LogMessageMonMB.genericTriggerEventPSet = genericTriggerEventFlag4fullTrackerAndHLTdb
