import FWCore.ParameterSet.Config as cms

l1tStage2CaloAnalyzer = cms.EDAnalyzer('L1TStage2CaloAnalyzer',
    mpBx = cms.int32(0),
    dmxBx = cms.int32(0),
    allBx = cms.bool(False),
    doEvtDisp = cms.bool(False),
    towerToken = cms.InputTag("caloStage2Digis", "CaloTower"), #After Compression
    towerPreCompressionToken = cms.InputTag("caloStage2Layer1Digis"), #Before Compression
    clusterToken = cms.InputTag("caloStage2Digis"),
    mpEGToken = cms.InputTag("caloStage2Digis", "MP"),
    mpTauToken = cms.InputTag("caloStage2Digis", "MP"),
    mpJetToken = cms.InputTag("caloStage2Digis", "MP"),
    mpEtSumToken = cms.InputTag("caloStage2Digis", "MP"),
    egToken = cms.InputTag("caloStage2Digis", "EGamma"),
    tauToken = cms.InputTag("caloStage2Digis","Tau"),
    jetToken = cms.InputTag("caloStage2Digis","Jet"),
    etSumToken = cms.InputTag("caloStage2Digis","EtSum")
)
