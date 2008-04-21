import FWCore.ParameterSet.Config as cms

# configuration of the alignment algorithm
#
LaserAlignmentAlgorithm = cms.PSet(
    AlignmentAlgorithm = cms.PSet(
        SecondFixedParameterTEC2TEC = cms.untracked.int32(3),
        FirstFixedParameterTEC2TEC = cms.untracked.int32(2),
        FirstFixedParameterNegTEC = cms.untracked.int32(2),
        SecondFixedParameterNegTEC = cms.untracked.int32(3),
        SecondFixedParameterPosTEC = cms.untracked.int32(3),
        # parameters to fix in Millepede (x,y for disc 1)
        FirstFixedParameterPosTEC = cms.untracked.int32(2)
    )
)

