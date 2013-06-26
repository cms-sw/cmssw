import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource")

process.demo = cms.EDAnalyzer('TestPSetAnalyzer',
    testLumi = cms.LuminosityBlockID("1:2"),
    testVLumi = cms.VLuminosityBlockID("1:2","2:2","3:3"),
    testRange = cms.LuminosityBlockRange("1:2-4:MAX"),
    testVRange = cms.VLuminosityBlockRange("1:2-4:MAX","99:99","3:4-5:9"),
    testERange = cms.EventRange("1:2-4:MAX"),
    testVERange = cms.VEventRange("1:2-4:MAX","99:99","3:4-5:9","9:9-11:MIN"),
    testEvent = cms.LuminosityBlockID("1:2")
)


process.p = cms.Path(process.demo)
