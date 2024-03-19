import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.MssmHbbBtagTriggerMonitor_cfi import *

mssmHbbBtagTriggerMonitor = cms.Sequence(
    mssmHbbBtagTriggerMonitorFH40  +
    mssmHbbBtagTriggerMonitorFH100 +
    mssmHbbBtagTriggerMonitorFH200 +
    mssmHbbBtagTriggerMonitorFH350 +
    mssmHbbBtagTriggerMonitorSL40  +
    mssmHbbBtagTriggerMonitorSL100 +
    mssmHbbBtagTriggerMonitorSL200 +
    mssmHbbBtagTriggerMonitorSL350
)
