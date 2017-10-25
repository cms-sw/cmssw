import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.JetMonitor_cff import *
from DQMOffline.Trigger.DiJetMonitor_cff import * 
jetmetMonitorHLT = cms.Sequence(
    HLTJetmonitoring *
    HLTDiJetmonitoring
)
