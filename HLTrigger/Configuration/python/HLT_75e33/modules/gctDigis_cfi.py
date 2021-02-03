import FWCore.ParameterSet.Config as cms

gctDigis = cms.EDProducer("GctRawToDigi",
    checkHeaders = cms.untracked.bool(False),
    gctFedId = cms.untracked.int32(745),
    hltMode = cms.bool(False),
    inputLabel = cms.InputTag("rawDataCollector"),
    mightGet = cms.optional.untracked.vstring,
    numberOfGctSamplesToUnpack = cms.uint32(1),
    numberOfRctSamplesToUnpack = cms.uint32(1),
    unpackSharedRegions = cms.bool(False),
    unpackerVersion = cms.uint32(0),
    verbose = cms.untracked.bool(False)
)
