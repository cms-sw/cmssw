
import FWCore.ParameterSet.Config as cms

l1tStage2InputPatternWriter = cms.EDAnalyzer('L1TStage2InputPatternWriter',
    towerToken     = cms.InputTag("caloStage2Layer1Digis"),
    filename       = cms.untracked.string("pattern.txt"),
    nChanPerQuad   = cms.untracked.int32(4),
    nQuads         = cms.untracked.int32(18),
    nHeaderFrames  = cms.untracked.int32(1),
    nPayloadFrames = cms.untracked.int32(39),
    nClearFrames   = cms.untracked.int32(6)
)
