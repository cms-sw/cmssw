import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

l1tTestsSummary = DQMEDHarvester("L1TTestsSummary",
  verbose = cms.untracked.bool(False),
  MonitorL1TRate      = cms.untracked.bool(True),
  MonitorL1TSync      = cms.untracked.bool(True),
  MonitorL1TOccupancy = cms.untracked.bool(False),
)
