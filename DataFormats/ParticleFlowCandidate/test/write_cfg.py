import FWCore.ParameterSet.Config as cms

process = cms.Process("WRITE")

process.source = cms.Source("EmptySource")

process.dummy = cms.EDProducer("TestDummyPFCandidateProducer")
process.tester = cms.EDAnalyzer("TestDummyPFCandidateAnalyzer", tag = cms.untracked.InputTag("dummy"))

process.p = cms.Path(process.dummy+process.tester)

process.out = cms.OutputModule("PoolOutputModule", fileName = cms.untracked.string("pfcand_test.root")
                               )
process.o = cms.EndPath(process.out)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10))
