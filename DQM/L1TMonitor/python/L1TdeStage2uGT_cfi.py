import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
l1tdeStage2uGT = DQMEDAnalyzer('L1TdeStage2uGT',
    dataSource = cms.InputTag("gtStage2Digis"),
    emulSource = cms.InputTag("valGtStage2Digis"),
    triggerBlackList = cms.vstring("L1_IsolatedBunch","*FirstBunch*","*SecondBunch*","*LastBunch*","*3BX","L1_CDC*"),
    numBxToMonitor = cms.int32(5),
    histFolder = cms.string('L1TEMU/L1TdeStage2uGT')
)
