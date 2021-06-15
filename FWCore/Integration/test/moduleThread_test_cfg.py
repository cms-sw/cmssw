import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")

process.source = cms.Source("EmptySource")

process.maxEvents.input = 10

process.test = cms.EDAnalyzer("edmtest::TestServicesOnNonFrameworkThreadsAnalyzer")

process.p = cms.EndPath(process.test)

process.add_(cms.Service("RandomNumberGeneratorService",
                         test = cms.PSet(initialSeed = cms.untracked.uint32(12345))
))
