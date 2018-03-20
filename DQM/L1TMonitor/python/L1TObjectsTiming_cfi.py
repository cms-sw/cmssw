import FWCore.ParameterSet.Config as cms

# the uGMT eta/phi map DQM module
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
l1tObjectsTiming = DQMEDAnalyzer(
    "L1TObjectsTiming",
    muonProducer  = cms.InputTag("gtStage2Digis", "Muon"),
    stage2CaloLayer2JetProducer = cms.InputTag("gtStage2Digis","Jet"),
    stage2CaloLayer2EGammaProducer = cms.InputTag("gtStage2Digis","EGamma"), 
    stage2CaloLayer2TauProducer = cms.InputTag("gtStage2Digis","Tau"),
    stage2CaloLayer2EtSumProducer = cms.InputTag("gtStage2Digis","EtSum"),
    ugtProducer = cms.InputTag("gtStage2Digis"),
    monitorDir = cms.untracked.string("L1T/L1TObjects"),
    verbose = cms.untracked.bool(False),
    firstBXInTrainAlgo = cms.untracked.string("L1_FirstCollisionInTrain"),
    lastBXInTrainAlgo = cms.untracked.string("L1_LastCollisionInTrain")
)

