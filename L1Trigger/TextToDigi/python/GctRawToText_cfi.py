import FWCore.ParameterSet.Config as cms

gctRawToText = cms.EDAnalyzer("RawToText",
    GctFedId = cms.untracked.int32(745),
    inputLabel = cms.InputTag("text2raw"),
    filename = cms.untracked.string('gctFromRaw.txt')
)


