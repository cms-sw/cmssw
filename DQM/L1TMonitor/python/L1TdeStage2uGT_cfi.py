import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
l1tdeStage2uGT = DQMEDAnalyzer('L1TdeStage2uGT',
    dataSource = cms.InputTag("gtStage2Digis"),
    emulSource = cms.InputTag("valGtStage2Digis"),
    triggerBlackList = cms.vstring("L1_IsolatedBunch","L1_FirstBunchInTrain","L1_FirstBunchAfterTrain","*3BX"),
    numBxToMonitor = cms.int32(5),
    histFolder = cms.string('L1TEMU/L1TdeStage2uGT')
)
