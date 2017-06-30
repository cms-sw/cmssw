import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.MssmHbbMonitor_cfi import *

mssmHbbMonitor = cms.Sequence(
    mssmHbbBtagTriggerMonitorSL40noMu +
    mssmHbbBtagTriggerMonitorSL40 +
    mssmHbbBtagTriggerMonitorSL100 +
    mssmHbbBtagTriggerMonitorSL200 +
    mssmHbbBtagTriggerMonitorSL350 +
    mssmHbbBtagTriggerMonitorAH100 +
    mssmHbbBtagTriggerMonitorAH200 +
    mssmHbbBtagTriggerMonitorAH350
)
