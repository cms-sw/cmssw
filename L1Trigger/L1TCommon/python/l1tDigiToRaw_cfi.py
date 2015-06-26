import FWCore.ParameterSet.Config as cms

l1tDigiToRaw = cms.EDProducer(
    "L1TDigiToRaw",
    Setup = cms.string("stage1::CaloSetup"),
    InputLabel = cms.InputTag("simCaloStage1FinalDigis"),
    TauInputLabel = cms.InputTag("simCaloStage1FinalDigis", "rlxTaus"),
    IsoTauInputLabel = cms.InputTag("simCaloStage1FinalDigis", "isoTaus"),
    HFBitCountsInputLabel = cms.InputTag("simCaloStage1FinalDigis", "HFBitCounts"),
    HFRingSumsInputLabel = cms.InputTag("simCaloStage1FinalDigis", "HFRingSums"),
    RegionInputLabel = cms.InputTag("simRctDigis", ""),
    EmCandInputLabel = cms.InputTag("simRctDigis", ""),
    FedId = cms.int32(1352),
    FWId = cms.uint32(1),
    lenSlinkHeader = cms.untracked.int32(8),
    lenSlinkTrailer = cms.untracked.int32(8)
)

