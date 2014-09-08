import FWCore.ParameterSet.Config as cms

caloStage1LegacyFormatDigis = cms.EDProducer(
    "l1t::L1TCaloUpgradeToGCTConverter",
    InputCollection = cms.InputTag("caloStage1FinalDigis")
)


