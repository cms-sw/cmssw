import FWCore.ParameterSet.Config as cms

l1tStage2CaloParams = cms.ESProducer(
    "L1TCaloESProducer",
    towerLsbH = cms.double32(0.5),
    towerLsbE = cms.double32(0.5),
    towerNBitsH = cms.int32(8),
    towerNBitsE = cms.int32(8)
)
