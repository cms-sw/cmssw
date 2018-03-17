import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
l1tStage2uGT = DQMEDAnalyzer('L1TStage2uGT',
    l1tStage2uGtSource = cms.InputTag("gtStage2Digis"),    
    monitorDir = cms.untracked.string("L1T/L1TStage2uGT"),
    verbose = cms.untracked.bool(False),
    firstBXInTrainAlgo = cms.untracked.string("L1_FirstCollisionInTrain"),
    lastBXInTrainAlgo = cms.untracked.string("L1_LastCollisionInTrain"),
    unprescaledAlgoShortList = cms.untracked.vstring("L1_SingleMu25","L1_SingleEG32","L1_DoubleMu_15_7","L1_SingleJet200","L1_HTT450er"),
    prescaledAlgoShortList = cms.untracked.vstring("L1_SingleMu3","L1_SingleEG15","L1_SingleJet90","L1_DoubleJet40er3p0","L1_HTT200er")
)
