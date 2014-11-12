import FWCore.ParameterSet.Config as cms

l1tDigiToRaw = cms.EDProducer(
    "l1t::L1TDigiToRaw",
    Setup = cms.string("stage2::CaloSetup"),
    InputLabel = cms.InputTag("caloStage2FinalDigis"),
    FedId = cms.int32(1300),
    FWId = cms.uint32(1)
)
