import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.MssmHbbBtagTriggerMonitor_cfi import *

mssmHbbBtagTriggerMonitor = cms.Sequence(
    mssmHbbDeepCSVBtagTriggerMonitorFH40  +
    mssmHbbDeepCSVBtagTriggerMonitorFH100 +
    mssmHbbDeepCSVBtagTriggerMonitorFH200 +
    mssmHbbDeepCSVBtagTriggerMonitorFH350 +
    mssmHbbDeepCSVBtagTriggerMonitorSL40  +
    mssmHbbDeepCSVBtagTriggerMonitorSL100 +
    mssmHbbDeepCSVBtagTriggerMonitorSL200 +
    mssmHbbDeepCSVBtagTriggerMonitorSL350 +
    mssmHbbDeepJetBtagTriggerMonitorFH40  +
    mssmHbbDeepJetBtagTriggerMonitorFH100 +
    mssmHbbDeepJetBtagTriggerMonitorFH200 +
    mssmHbbDeepJetBtagTriggerMonitorFH350 +
    mssmHbbDeepJetBtagTriggerMonitorSL40  +
    mssmHbbDeepJetBtagTriggerMonitorSL100 +
    mssmHbbDeepJetBtagTriggerMonitorSL200 +
    mssmHbbDeepJetBtagTriggerMonitorSL350
)
