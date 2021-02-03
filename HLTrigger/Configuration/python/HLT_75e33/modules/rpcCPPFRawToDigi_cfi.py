import FWCore.ParameterSet.Config as cms

rpcCPPFRawToDigi = cms.EDProducer("RPCAMCRawToDigi",
    RPCAMCUnpacker = cms.string('RPCCPPFUnpacker'),
    RPCAMCUnpackerSettings = cms.PSet(
        bxMax = cms.int32(2),
        bxMin = cms.int32(-2),
        fillAMCCounters = cms.bool(True)
    ),
    calculateCRC = cms.bool(True),
    fillCounters = cms.bool(True),
    inputTag = cms.InputTag("rawDataCollector")
)
