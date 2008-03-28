import FWCore.ParameterSet.Config as cms

ecalBarrelMonitorModule = cms.EDFilter("EcalBarrelMonitorModule",
    EcalRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    verbose = cms.untracked.bool(False),
    EBDigiCollection = cms.InputTag("ecalEBunpacker","ebDigis"),
    EcalTBEventHeader = cms.InputTag("ecalEBunpacker"),
    enableEventDisplay = cms.untracked.bool(False),
    enableCleanup = cms.untracked.bool(False),
    EcalRawDataCollection = cms.InputTag("ecalEBunpacker"),
    runNumber = cms.untracked.int32(0),
    runType = cms.untracked.int32(-1),
    EcalTrigPrimDigiCollection = cms.InputTag("ecalEBunpacker","EcalTriggerPrimitives")
)


