import FWCore.ParameterSet.Config as cms

caloStage1Digis = cms.EDProducer(
    "L1TRawToDigi",
    Setup = cms.string("stage1::CaloSetup"),
    #InputLabel = cms.InputTag("l1tDigiToRaw"),
    InputLabel = cms.InputTag("rawDataCollector"),
    FedId = cms.int32(1352),
    FWId = cms.untracked.int32(2)
)
