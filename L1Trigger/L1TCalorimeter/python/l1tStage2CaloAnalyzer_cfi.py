import FWCore.ParameterSet.Config as cms

l1tStage2CaloAnalyzer = cms.EDAnalyzer('L1TStage2CaloAnalyzer',
    towerToken = cms.InputTag("caloStage2Digis"), #After Compression
    towerPreCompressionToken = cms.InputTag("caloStage2Layer1Digis"), #Before Compression
    clusterToken = cms.InputTag("caloStage2Digis"),
    mpEGToken = cms.InputTag("caloStage2Digis", "MP"),
    mpTauToken = cms.InputTag("caloStage2Digis", "MP"),
    mpJetToken = cms.InputTag("caloStage2Digis", "MP"),
    mpEtSumToken = cms.InputTag("caloStage2Digis", "MP"),
    egToken = cms.InputTag("caloStage2Digis"),
    tauToken = cms.InputTag("caloStage2Digis"),
    jetToken = cms.InputTag("caloStage2Digis"),
    etSumToken = cms.InputTag("caloStage2Digis")
)
