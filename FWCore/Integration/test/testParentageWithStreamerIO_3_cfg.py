import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST3")

process.source = cms.Source("NewEventStreamFileReader",
    fileNames = cms.untracked.vstring('file:testParentageWithStreamerIO2.dat')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.test1 = cms.EDAnalyzer("TestParentage",
    inputTag = cms.InputTag("prod2"),
    expectedAncestors = cms.vstring()
)

process.test2 = cms.EDAnalyzer("TestParentage",
    inputTag = cms.InputTag("prod12"),
    expectedAncestors = cms.vstring("prod11")
)

process.path1 = cms.Path(process.test1 + process.test2)
