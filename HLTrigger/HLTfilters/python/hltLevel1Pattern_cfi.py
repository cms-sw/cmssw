import FWCore.ParameterSet.Config as cms

hltLevel1Pattern = cms.EDFilter("HLTLevel1Pattern",
    L1GtReadoutRecordTag  = cms.InputTag('gtDigis'),
    triggerBit     = cms.string( 'L1Tech_BPTX_plus_AND_minus.v0' ),
    triggerPattern = cms.vint32( 0,  0,  1,  0,  0),
    bunchCrossings = cms.vint32(-2, -1,  0,  1,  2),
    ignoreL1Mask   = cms.bool( False ),
    invert         = cms.bool( False )
)
