import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.MssmHbbMonitor_cfi import *

mssmHbbMonitor = cms.Sequence(
    mssmHbbBtagTriggerMonitor
)
