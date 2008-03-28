import FWCore.ParameterSet.Config as cms

#Common Parameters for the various CSCTrigger producers
CSCCommonTrigger = cms.PSet(
    MaxBX = cms.int32(9),
    #minimum and maximum bx to process
    MinBX = cms.int32(3)
)

