import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.VBFMETMonitor_cff import *

higgsMonitorHLT = cms.Sequence(
    higgsinvHLTJetMETmonitoring
)
