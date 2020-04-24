import FWCore.ParameterSet.Config as cms

caloLayer1Digis = cms.EDProducer(
    "L1TRawToDigi",
    Setup = cms.string("stage2::CaloLayer1Setup"),
    InputLabel = cms.InputTag("rawDataCollector"),
    FedIds = cms.vint32(1354, 1356, 1358),
    FWId = cms.uint32(0x12345678),
    debug = cms.untracked.bool(False),
    CTP7 = cms.untracked.bool(True)
)

