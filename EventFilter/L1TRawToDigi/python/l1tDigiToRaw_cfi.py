import FWCore.ParameterSet.Config as cms

l1tDigiToRaw = cms.EDProducer(
    "L1TDigiToRaw",
    Setup = cms.string("stage2::CaloSetup"),
    InputLabel = cms.InputTag("caloStage2FinalDigis"),
    FedId = cms.int32(1),
    FWId = cms.uint32(1)
)
