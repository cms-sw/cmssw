import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.MssmHbbMonitor_cfi import *

mssmHbbMonitor = cms.Sequence(
    msssHbbAllHadronic100 +
    msssHbbAllHadronic116 +
    msssHbbAllHadronic128 +
    msssHbbSemileptonic40 +
    msssHbbSemileptonic54 +
    msssHbbSemileptonic62 +
    msssHbbSemileptonicNoBtag +
    mssmHbbBtagTriggerMonitorSL40noMu +
    mssmHbbBtagTriggerMonitorSL40 +
    mssmHbbBtagTriggerMonitorSL100 +
    mssmHbbBtagTriggerMonitorSL200 +
    mssmHbbBtagTriggerMonitorSL350 +
    mssmHbbBtagTriggerMonitorAH100 +
    mssmHbbBtagTriggerMonitorAH200 +
    mssmHbbBtagTriggerMonitorAH350
)
