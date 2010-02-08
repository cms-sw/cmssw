import FWCore.ParameterSet.Config as cms

hltLevel1Activity = cms.EDFilter("HLTLevel1Activity",
    L1GtReadoutRecordTag  = cms.InputTag('gtDigis'),
    bunchCrossings = cms.vint32( 0, -1, 1),    # BPTX +/- 1 
    ignoreL1Mask   = cms.bool( False ),        # use L1 masks
    physicsLoBits  = cms.uint64( 0xFFFFFFFE ), # all physics bits except BPTX (L1_ZeroBias, bit 0)
    physicsHiBits  = cms.uint64( 0xFFFFFFFF ),
    technicalBits  = cms.uint64( 0xFFFFFF00 )  # all technical bits except BPTX (bits 0-7)
)
