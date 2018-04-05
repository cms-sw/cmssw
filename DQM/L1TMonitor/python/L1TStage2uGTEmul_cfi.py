import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
l1tStage2uGtEmul = DQMEDAnalyzer('L1TStage2uGT',
    l1tStage2uGtSource = cms.InputTag("valGtStage2Digis"),    
    monitorDir = cms.untracked.string("L1TEMU/L1TStage2uGTEmul"),
    verbose = cms.untracked.bool(False)
)
