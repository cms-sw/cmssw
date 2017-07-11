import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.VBFMETMonitor_Client_cff import *
from DQMOffline.Trigger.MssmHbbBtagTriggerMonitor_Client_cfi import *
from DQMOffline.Trigger.MssmHbbMonitoring_Client_cfi import *

higgsClient = cms.Sequence(
    vbfmetClient
    mssmHbbBtagTriggerEfficiency +
    mssmHbbHLTEfficiency +
)
