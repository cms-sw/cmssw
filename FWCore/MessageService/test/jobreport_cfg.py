import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.maxEvents.input = 1
process.source = cms.Source("EmptySource")
process.a = cms.EDAnalyzer("edmtest::PrintProcessInformation")
process.p = cms.Path(process.a)
