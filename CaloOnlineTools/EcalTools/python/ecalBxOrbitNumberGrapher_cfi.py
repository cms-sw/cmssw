import FWCore.ParameterSet.Config as cms

ecalBxOrbitNumberGrapher = cms.EDAnalyzer("EcalBxOrbitNumberGrapher",

    EBRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    EERecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),

    RawDigis = cms.string('ecalEBunpacker'),

    # parameter to specify histogram maxmimum range
    histogramMaxRange = cms.untracked.double(200.0),

    # parameter to specify histogram minimum range
    histogramMinRange = cms.untracked.double(-10.0),

    # parameter for the name of the output root file with TH1F
    fileName = cms.untracked.string('bxOrbitHists')

)
