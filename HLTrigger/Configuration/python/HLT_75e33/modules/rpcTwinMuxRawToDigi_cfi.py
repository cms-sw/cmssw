import FWCore.ParameterSet.Config as cms

rpcTwinMuxRawToDigi = cms.EDProducer("RPCTwinMuxRawToDigi",
    bxMax = cms.int32(2),
    bxMin = cms.int32(-2),
    calculateCRC = cms.bool(True),
    fillCounters = cms.bool(True),
    inputTag = cms.InputTag("rawDataCollector"),
    mightGet = cms.optional.untracked.vstring
)
