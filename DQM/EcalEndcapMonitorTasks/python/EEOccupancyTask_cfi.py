import FWCore.ParameterSet.Config as cms

ecalEndcapOccupancyTask = cms.EDAnalyzer("EEOccupancyTask",
    prefixME = cms.untracked.string('EcalEndcap'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),
                                         subfolder = cms.untracked.string(''),
    EcalRawDataCollection = cms.InputTag("ecalDigis"),
    EEDigiCollection = cms.InputTag("ecalDigis","eeDigis"),
    EcalTrigPrimDigiCollection = cms.InputTag("ecalDigis","EcalTriggerPrimitives"),
    EcalPnDiodeDigiCollection = cms.InputTag("ecalDigis"),
    EcalRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE")
)

