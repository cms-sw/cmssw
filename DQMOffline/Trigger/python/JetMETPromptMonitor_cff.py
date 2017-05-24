import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.JetMETMonitor_cff import *

jetmetMonitorHLT = cms.Sequence(
    HLTJetMETmonitoring
)
