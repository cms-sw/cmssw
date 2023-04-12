import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")

process.source = cms.Source("EmptySource")

process.maxEvents.input = 1

process.add_(cms.Service("Timing", summaryOnly = cms.untracked.bool(True)))