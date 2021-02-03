import FWCore.ParameterSet.Config as cms

caloStage1Digis = cms.EDProducer("L1TRawToDigi",
    FedIds = cms.vint32(1352),
    InputLabel = cms.InputTag("rawDataCollector"),
    Setup = cms.string('stage1::CaloSetup')
)
