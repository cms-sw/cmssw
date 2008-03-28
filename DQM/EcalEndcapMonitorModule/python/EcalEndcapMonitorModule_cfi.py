import FWCore.ParameterSet.Config as cms

ecalEndcapMonitorModule = cms.EDFilter("EcalEndcapMonitorModule",
    EcalRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    EEDigiCollection = cms.InputTag("ecalEBunpacker","eeDigis"),
    verbose = cms.untracked.bool(False),
    EcalTBEventHeader = cms.InputTag("ecalEBunpacker"),
    enableEventDisplay = cms.untracked.bool(False),
    enableCleanup = cms.untracked.bool(False),
    EcalRawDataCollection = cms.InputTag("ecalEBunpacker"),
    runNumber = cms.untracked.int32(0),
    runType = cms.untracked.int32(-1),
    EcalTrigPrimDigiCollection = cms.InputTag("ecalEBunpacker","EcalTriggerPrimitives")
)


