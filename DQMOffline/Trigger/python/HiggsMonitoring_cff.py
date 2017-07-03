import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.MssmHbbBtagTriggerMonitor_cff import *

higgsMonitorHLT = cms.Sequence(
   mssmHbbBtagTriggerMonitor
)
