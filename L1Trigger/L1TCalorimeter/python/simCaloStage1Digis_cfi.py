import FWCore.ParameterSet.Config as cms

simCaloStage1Digis = cms.EDProducer(
    "L1TStage1Layer2Producer",
    CaloRegions = cms.InputTag("simRctUpgradeFormatDigis"),
    CaloEmCands = cms.InputTag("simRctUpgradeFormatDigis"),
    conditionsLabel = cms.string("")
)
