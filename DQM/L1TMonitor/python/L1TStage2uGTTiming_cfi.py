import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

l1tStage2uGTTiming = DQMEDAnalyzer('L1TStage2uGTTiming',
    l1tStage2uGtSource = cms.InputTag("gtStage2Digis"),    
    monitorDir = cms.untracked.string("L1T/L1TStage2uGT/timing_aux"),
    verbose = cms.untracked.bool(False),
    firstBXInTrainAlgo = cms.untracked.string("L1_FirstCollisionInTrain"),
    lastBXInTrainAlgo = cms.untracked.string("L1_LastCollisionInTrain"),    
    isoBXAlgo = cms.untracked.string("L1_IsolatedBunch"),    
    unprescaledAlgoShortList = cms.untracked.vstring(
        "L1_SingleMu22_BMTF",
        "L1_SingleMu22_OMTF",
        "L1_SingleMu22_EMTF",
        "L1_SingleIsoEG28er1p5",
        "L1_SingleIsoEG32er2p5",
        "L1_SingleEG40er2p5",
        "L1_SingleEG60",
        "L1_SingleTau120er2p1",
        "L1_SingleJet180",
        "L1_ETMHF130",
        "L1_HTT360er",
        "L1_ETT2000"
    ),
    prescaledAlgoShortList = cms.untracked.vstring(
        "L1_FirstCollisionInTrain",
        "L1_LastCollisionInTrain",
        "L1_IsolatedBunch",
        "L1_SingleMu0_BMTF",
        "L1_SingleMu0_OMTF",
        "L1_SingleMu0_EMTF",
        "L1_SingleEG10er2p5",
        "L1_SingleEG15er2p5",
        "L1_SingleEG26er2p5",
        "L1_SingleLooseIsoEG28er1p5",
        "L1_SingleJet60",
        "L1_SingleJet60er2p5",
        "L1_SingleJet60_FWD3p0",
        "L1_ETMHF100",
        "L1_HTT120er",
        "L1_ETT1200"
    ),
    useAlgoDecision = cms.untracked.string("initial")
)
