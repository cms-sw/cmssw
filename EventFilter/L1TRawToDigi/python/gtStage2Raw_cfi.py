import FWCore.ParameterSet.Config as cms

gtStage2Raw = cms.EDProducer(
    "L1TDigiToRaw",
    Setup = cms.string("stage2::GTSetup"),
#    TowerInputLabel = cms.InputTag("simCaloStage2Digis"),
    GtInputTag = cms.InputTag("simGtStage2Digis"),
    ExtInputTag = cms.InputTag("simGtStage2Digis"),
    MuonInputTag   = cms.InputTag("simGmtStage2Digis"),
    EGammaInputTag = cms.InputTag("simCaloStage2Digis"),
    TauInputTag    = cms.InputTag("simCaloStage2Digis"),
    JetInputTag    = cms.InputTag("simCaloStage2Digis"),
    EtSumInputTag  = cms.InputTag("simCaloStage2Digis"),
    FedId = cms.int32(1404),
    FWId = cms.uint32(0x10A5), # FIXME: Set correct FW version for switch to 2017 muon RAW format (0x10A4 was used in run 287320 (MWGR#1))
    lenSlinkHeader = cms.untracked.int32(8),
    lenSlinkTrailer = cms.untracked.int32(8)
)
