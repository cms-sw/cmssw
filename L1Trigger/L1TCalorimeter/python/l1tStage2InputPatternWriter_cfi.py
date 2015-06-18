
import FWCore.ParameterSet.Config as cms

l1tStage2InputPatternWriter = cms.EDAnalyzer('L1TStage2InputPatternWriter',
    towerToken     = cms.InputTag("caloStage2Layer1Digis"),
    filename       = cms.untracked.string("pattern.txt"),
    nChanPerQuad   = cms.untracked.uint32(4),
    nQuads         = cms.untracked.uint32(18),
    nHeaderFrames  = cms.untracked.uint32(1),
    nPayloadFrames = cms.untracked.uint32(39),
    nClearFrames   = cms.untracked.uint32(6)
)
