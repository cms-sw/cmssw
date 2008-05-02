import FWCore.ParameterSet.Config as cms

ecalEndcapMonitorModule = cms.EDFilter("EcalEndcapMonitorModule",
    EcalRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    EEDigiCollection = cms.InputTag("ecalEBunpacker","eeDigis"),
    verbose = cms.untracked.bool(True),
    runNumber = cms.untracked.int32(0),
    enableEventDisplay = cms.untracked.bool(False),
    enableCleanup = cms.untracked.bool(False),
    debug = cms.untracked.bool(False),
    EcalTrigPrimDigiCollection = cms.InputTag("ecalEBunpacker","EcalTriggerPrimitives"),
    EcalRawDataCollection = cms.InputTag("ecalEBunpacker"),
    runType = cms.untracked.int32(-1),
    prefixME = cms.untracked.string('EcalEndcap')
)


