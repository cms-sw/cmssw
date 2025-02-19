import FWCore.ParameterSet.Config as cms

ecalPulseShapeGrapher = cms.EDAnalyzer("EcalPulseShapeGrapher",
    EBDigiCollection = cms.InputTag("ecalEBunpacker","ebDigis"),
    EEDigiCollection = cms.InputTag("ecalEBunpacker","eeDigis"),
    EBUncalibratedRecHitCollection = cms.InputTag("ecalUncalibHit","EcalUncalibRecHitsEB"),
    EEUncalibratedRecHitCollection = cms.InputTag("ecalUncalibHit","EcalUncalibRecHitsEE"),

    listChannels = cms.untracked.vint32(42803,28472,49867,8108,56473,53845,1896,56264,53316,53677,53678),
    AmplitudeCutADC = cms.untracked.int32(13),
    rootFilename = cms.untracked.string('ecalPulseShapeGrapher')
)
