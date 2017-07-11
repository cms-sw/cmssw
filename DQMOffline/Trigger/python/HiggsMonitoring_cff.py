import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.VBFMETMonitor_cff import *
from DQMOffline.Trigger.MssmHbbBtagTriggerMonitor_cff import *
from DQMOffline.Trigger.MssmHbbMonitoring_cff import *

higgsMonitorHLT = cms.Sequence(

    higgsinvHLTJetMETmonitoring
    mssmHbbBtagTriggerMonitor +
    mssmHbbMonitorHLT +
)

