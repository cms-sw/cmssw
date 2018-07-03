import FWCore.ParameterSet.Config as cms

rpcCPPFRawToDigi = cms.EDProducer('RPCAMCRawToDigi',
    inputTag = cms.InputTag('rawDataCollector'),
    calculateCRC = cms.bool(True),
    fillCounters = cms.bool(True),
    RPCAMCUnpacker = cms.string('RPCCPPFUnpacker'),
    RPCAMCUnpackerSettings = cms.PSet(
        fillAMCCounters = cms.bool(True),
        bxMin = cms.int32(-2),
        bxMax = cms.int32(2)
    )
)
