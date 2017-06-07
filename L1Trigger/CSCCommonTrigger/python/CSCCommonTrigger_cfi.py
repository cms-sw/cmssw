import FWCore.ParameterSet.Config as cms

#Common Parameters for the various CSCTrigger producers
CSCCommonTrigger = cms.PSet(
    #shift the readout window from [3, 9] to [5,11], make sure simulation consistent with data, by tao.huang@cern.ch
    MaxBX = cms.int32(11),
    #minimum and maximum bx to process
    MinBX = cms.int32(5)
)

