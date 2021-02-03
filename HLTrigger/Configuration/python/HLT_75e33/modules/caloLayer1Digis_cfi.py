import FWCore.ParameterSet.Config as cms

caloLayer1Digis = cms.EDProducer("L1TRawToDigi",
    CTP7 = cms.untracked.bool(True),
    FWId = cms.uint32(305419896),
    FedIds = cms.vint32(1354, 1356, 1358),
    InputLabel = cms.InputTag("rawDataCollector"),
    Setup = cms.string('stage2::CaloLayer1Setup'),
    debug = cms.untracked.bool(False)
)
