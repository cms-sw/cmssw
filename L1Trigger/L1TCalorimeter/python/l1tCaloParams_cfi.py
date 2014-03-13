import FWCore.ParameterSet.Config as cms

l1tCaloParamsSource = cms.ESSource(
    "EmptyESSource",
    recordName = cms.string('L1TCaloParamsRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

l1tStage2CaloParams = cms.ESProducer(
    "l1t::L1TCaloParamsESProducer",
    firmwarePP = cms.int32(1),
    firmwareMP = cms.int32(1),
    towerLsbH = cms.double(0.5),
    towerLsbE = cms.double(0.5),
    towerNBitsH = cms.int32(8),
    towerNBitsE = cms.int32(8)
)
