import FWCore.ParameterSet.Config as cms

gtStage2Raw = cms.EDProducer(
    "L1TDigiToRaw",
    Setup = cms.string("stage2::GTSetup"),
#    TowerInputLabel = cms.InputTag("simCaloStage2Digis"),
    GtInputLabel = cms.InputTag("simGtStage2Digis"),
    ExtInputLabel = cms.InputTag("simGtStage2Digis"),
    CaloInputLabel = cms.InputTag("simCaloStage2Digis"),
    GmtInputLabel = cms.InputTag("simGmtStage2Digis"),
    FedId = cms.int32(1404),
    FWId = cms.uint32(1),
    lenSlinkHeader = cms.untracked.int32(8),
    lenSlinkTrailer = cms.untracked.int32(8)
)
