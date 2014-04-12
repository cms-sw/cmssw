import FWCore.ParameterSet.Config as cms

l1tTestsSummary = cms.EDAnalyzer("L1TTestsSummary",
  verbose = cms.untracked.bool(False),
  MonitorL1TRate      = cms.untracked.bool(True),
  MonitorL1TSync      = cms.untracked.bool(True),
  MonitorL1TOccupancy = cms.untracked.bool(False),
)