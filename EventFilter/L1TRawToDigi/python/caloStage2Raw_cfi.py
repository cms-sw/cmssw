import FWCore.ParameterSet.Config as cms

caloStage2Raw = cms.EDProducer(
    "L1TDigiToRaw",
    Setup = cms.string("stage2::CaloSetup"),
    TowerInputLabel = cms.InputTag("simCaloStage2Layer1Digis"),
    InputLabel = cms.InputTag("simCaloStage2Digis"),
    FedId = cms.int32(1366),
    FWId = cms.uint32(1),
    lenSlinkHeader = cms.untracked.int32(8),
    lenSlinkTrailer = cms.untracked.int32(8)
)
