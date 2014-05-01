import FWCore.ParameterSet.Config as cms

l1tStage2CaloAnalyzer = cms.EDAnalyzer('l1t::Stage2CaloAnalyzer',
    towerToken = cms.InputTag("l1tCaloStage2Digis"), #After Compression
    towerPreCompressionToken = cms.InputTag("l1tCaloStage2Layer1Digis"), #Before Compression
    clusterToken = cms.InputTag("l1tCaloStage2Digis"),
    egToken = cms.InputTag("l1tCaloStage2Digis"),
    tauToken = cms.InputTag("l1tCaloStage2Digis"),
    jetToken = cms.InputTag("l1tCaloStage2Digis"),
    etSumToken = cms.InputTag("l1tCaloStage2Digis")
)
