
import FWCore.ParameterSet.Config as cms

l1tStage2InputPatternWriter = cms.EDAnalyzer('Stage2InputPatternWriter',
    towerToken = cms.InputTag("caloStage2Layer1Digis"),
    filename   = cms.untracked.string("pattern.txt")
)
