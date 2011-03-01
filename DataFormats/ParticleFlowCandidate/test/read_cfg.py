import FWCore.ParameterSet.Config as cms

process = cms.Process("READ")

process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring("file:pfcand_test.root"))

process.tester = cms.EDAnalyzer("TestDummyPFCandidateAnalyzer", tag = cms.untracked.InputTag("dummy"))

process.p = cms.Path(process.tester)

