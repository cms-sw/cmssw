import FWCore.ParameterSet.Config as cms

ecalBarrelOccupancyTask = cms.EDAnalyzer("EBOccupancyTask",
    prefixME = cms.untracked.string('EcalBarrel'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),
                                         subfolder = cms.untracked.string(''),
    EcalRawDataCollection = cms.InputTag("ecalDigis"),
    EBDigiCollection = cms.InputTag("ecalDigis","ebDigis"),
    EcalTrigPrimDigiCollection = cms.InputTag("ecalDigis","EcalTriggerPrimitives"),
    EcalPnDiodeDigiCollection = cms.InputTag("ecalDigis"),
    EcalRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB")
)

