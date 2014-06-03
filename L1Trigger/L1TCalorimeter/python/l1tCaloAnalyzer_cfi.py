import FWCore.ParameterSet.Config as cms

l1tStage2CaloAnalyzer = cms.EDAnalyzer('l1t::Stage2CaloAnalyzer',
    towerToken = cms.InputTag("caloStage2Digis"), #After Compression
    towerPreCompressionToken = cms.InputTag("caloStage2Layer1Digis"), #Before Compression
    clusterToken = cms.InputTag("caloStage2Digis"),
    egToken = cms.InputTag("caloStage2Digis"),
    tauToken = cms.InputTag("caloStage2Digis"),
    jetToken = cms.InputTag("caloStage2Digis"),
    etSumToken = cms.InputTag("caloStage2Digis")
)
