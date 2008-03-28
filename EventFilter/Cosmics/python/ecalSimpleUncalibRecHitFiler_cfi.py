import FWCore.ParameterSet.Config as cms

ecalSimpleUncalibRecHitFiler = cms.EDFilter("EcalSimpleUncalibRecHitFilter",
    adcCut = cms.untracked.double(12.0),
    EcalUncalibRecHitCollection = cms.InputTag("ecalWeightUncalibRecHit","EcalUncalibRecHitsEB"),
    #
    maskedChannels = cms.untracked.vint32()
)


