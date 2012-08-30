import FWCore.ParameterSet.Config as cms

ecalBarrelMonitorModule = cms.EDAnalyzer("EcalBarrelMonitorModule",
    EcalRawDataCollection = cms.InputTag("ecalDigis"),
    EBDigiCollection = cms.InputTag("ecalDigis","ebDigis"),
    EcalRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    EcalTrigPrimDigiCollection = cms.InputTag("ecalDigis","EcalTriggerPrimitives"),
    prefixME = cms.untracked.string('EcalBarrel'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),
    enableEventDisplay = cms.untracked.bool(False),
    runNumber = cms.untracked.int32(0),
    runType = cms.untracked.int32(-1),
    verbose = cms.untracked.bool(True),
    debug = cms.untracked.bool(False)
)

