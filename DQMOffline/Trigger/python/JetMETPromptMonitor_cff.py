import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.JetMonitor_cff import *

jetmetMonitorHLT = cms.Sequence(
    HLTJetmonitoring
)
