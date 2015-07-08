import FWCore.ParameterSet.Config as cms

caloStage1Raw = cms.EDProducer(
    "L1TDigiToRaw",
    Setup = cms.string("stage1::CaloSetup"),
    InputLabel = cms.InputTag("caloStage1Digis"),
    TauInputLabel = cms.InputTag("caloStage1Digis", "rlxTaus"),
    IsoTauInputLabel = cms.InputTag("caloStage1Digis", "isoTaus"),
    HFBitCountsInputLabel = cms.InputTag("caloStage1Digis", "HFBitCounts"),
    HFRingSumsInputLabel = cms.InputTag("caloStage1Digis", "HFRingSums"),
    RegionInputLabel = cms.InputTag("caloStage1Digis", ""),
    EmCandInputLabel = cms.InputTag("caloStage1Digis", ""),
    FedId = cms.int32(1352),
    FWId = cms.uint32(1),
    lenSlinkHeader = cms.untracked.int32(8),
    lenSlinkTrailer = cms.untracked.int32(8)
)
