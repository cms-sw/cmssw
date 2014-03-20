import FWCore.ParameterSet.Config as cms

l1tCaloAnalyzer = cms.EDAnalyzer('L1TCaloAnalyzer',
    towerToken = cms.InputTag("l1tCaloStage2TowerDigis"),
    clusterToken = cms.InputTag("l1tCaloStage2Digis"),
    egToken = cms.InputTag("l1tCaloStage2Digis"),
    tauToken = cms.InputTag("l1tCaloStage2Digis"),
    jetToken = cms.InputTag("l1tCaloStage2Digis"),
    etSumToken = cms.InputTag("l1tCaloStage2Digis")
)
