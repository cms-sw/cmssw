import FWCore.ParameterSet.Config as cms

ecalPrescaler = cms.EDFilter("EcalMonitorPrescaler",
    EcalRawDataCollection = cms.InputTag("ecalDigis"),
    occupancyPrescaleFactor = cms.untracked.int32(1),
    integrityPrescaleFactor = cms.untracked.int32(1),
    cosmicPrescaleFactor = cms.untracked.int32(1),
    laserPrescaleFactor = cms.untracked.int32(1),
    ledPrescaleFactor = cms.untracked.int32(1),
    pedestalonlinePrescaleFactor = cms.untracked.int32(1),
    pedestalPrescaleFactor = cms.untracked.int32(1),
    testpulsePrescaleFactor = cms.untracked.int32(1),
    pedestaloffsetPrescaleFactor = cms.untracked.int32(1),
    triggertowerPrescaleFactor = cms.untracked.int32(1),
    timingPrescaleFactor = cms.untracked.int32(1),
    physicsPrescaleFactor = cms.untracked.int32(1),
    clusterPrescaleFactor = cms.untracked.int32(1)
)

