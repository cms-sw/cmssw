import FWCore.ParameterSet.Config as cms

ecalURecHitHists = cms.EDAnalyzer("EcalURecHitHists",
    EBDigiCollection = cms.InputTag("ecalEBunpacker","ebDigis"),
    EBUncalibratedRecHitCollection = cms.InputTag("ecalUncalibHit","EcalUncalibRecHitsEB"),
    EEUncalibratedRecHitCollection = cms.InputTag("ecalUncalibHit","EcalUncalibRecHitsEE"),
    # use hash index to mask channels
    # add a simple description of hashIndex (hhahhahhh...)
    maskedChannels = cms.untracked.vint32(),
    # masked FEDs
    #untracked vint32 maskedFEDs = {-1}
    # masked EBids
    #untracked vstring maskedEBs = {"-1"}
    # parameter to specify histogram maxmimum range
    #untracked double histogramMaxRange = 200.0
    # parameter to specify histogram minimum range
    #untracked double histogramMinRange = -10.0
    # parameter for the name of the output root file with TH1F
    fileName = cms.untracked.string('ecalURecHitHists')
)


