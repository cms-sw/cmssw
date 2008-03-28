import FWCore.ParameterSet.Config as cms

ecalPrescaler = cms.EDFilter("EcalMonitorPrescaler",
    pedestalonlinePrescaleFactor = cms.untracked.int32(1),
    occupancyPrescaleFactor = cms.untracked.int32(1),
    pedestalPrescaleFactor = cms.untracked.int32(1),
    clusterPrescaleFactor = cms.untracked.int32(1),
    laserPrescaleFactor = cms.untracked.int32(1),
    EcalRawDataCollection = cms.InputTag("ecalEBunpacker"),
    integrityPrescaleFactor = cms.untracked.int32(1),
    testpulsePrescaleFactor = cms.untracked.int32(1),
    triggertowerPrescaleFactor = cms.untracked.int32(1),
    timingPrescaleFactor = cms.untracked.int32(1),
    cosmicPrescaleFactor = cms.untracked.int32(1)
)


