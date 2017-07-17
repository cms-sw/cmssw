import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

process.stuck = cms.EDAnalyzer("StuckAnalyzer")
process.p = cms.Path(process.stuck)

process.add_(cms.Service("ZombieKillerService",
                          secondsBetweenChecks = cms.untracked.uint32(10),
                          numberOfAllowedFailedChecksInARow = cms.untracked.uint32(2)))
