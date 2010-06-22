import FWCore.ParameterSet.Config as cms

process = cms.Process("WriteTest")
process.source = cms.Source( "EmptySource")
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.writer = cms.EDAnalyzer("FWLiteESRecordWriterAnalyzer",
                                fileName = cms.untracked.string("cond_test.root"),
                                IntProductRecord = cms.untracked.VPSet(cms.untracked.PSet(type=cms.untracked.string("edmtest::IntProduct")))
)

process.add_(cms.ESSource("IntProductESSource"))

process.out = cms.EndPath(process.writer)

