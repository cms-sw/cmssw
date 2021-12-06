import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")

process.source = cms.Source("EmptySource")

process.maxEvents.input = 1

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.INFO.limit = cms.untracked.int32(1000000000)

process.dummyAna = cms.EDAnalyzer("TTTrackTrackWordDummyOneAnalyzer")

process.p = cms.Path(process.dummyAna)