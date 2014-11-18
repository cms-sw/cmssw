import FWCore.ParameterSet.Config as cms

l1tDigiToRaw = cms.EDProducer(
    "L1TDigiToRaw",
    Setup = cms.string("stage1::CaloSetup"),
    InputLabel = cms.InputTag("caloStage2FinalDigis"),
    FedId = cms.int32(1300),
    FWId = cms.uint32(1)
)
