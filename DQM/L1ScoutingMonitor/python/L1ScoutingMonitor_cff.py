import FWCore.ParameterSet.Config as cms
from DQM.L1ScoutingMonitor.L1ScoutingMonitor_cfi import L1ScoutingMonitor

l1sMonitorSequence = cms.Sequence(L1ScoutingMonitor)
