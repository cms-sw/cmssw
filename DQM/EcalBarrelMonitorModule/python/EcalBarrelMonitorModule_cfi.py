import FWCore.ParameterSet.Config as cms

ecalBarrelMonitorModule = cms.EDFilter("EcalBarrelMonitorModule",
    EcalRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    verbose = cms.untracked.bool(True),
    EBDigiCollection = cms.InputTag("ecalEBunpacker","ebDigis"),
    runNumber = cms.untracked.int32(0),
    enableEventDisplay = cms.untracked.bool(False),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),
    debug = cms.untracked.bool(False),
    EcalTrigPrimDigiCollection = cms.InputTag("ecalEBunpacker","EcalTriggerPrimitives"),
    EcalRawDataCollection = cms.InputTag("ecalEBunpacker"),
    runType = cms.untracked.int32(-1),
    prefixME = cms.untracked.string('EcalBarrel')
)


