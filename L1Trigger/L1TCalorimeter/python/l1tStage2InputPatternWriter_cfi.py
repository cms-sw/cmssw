
import FWCore.ParameterSet.Config as cms

l1tStage2InputPatternWriter = cms.EDAnalyzer('Stage2InputPatternWriter',
    towerToken = cms.InputTag("l1tCaloStage2Layer1Digis"),
    filename   = cms.untracked.string("pattern.txt")
)
