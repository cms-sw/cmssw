import FWCore.ParameterSet.Config as cms

# producer of rechits starting from digis
ecalFixedAlphaBetaFitUncalibRecHit = cms.EDProducer("EcalFixedAlphaBetaFitUncalibRecHitProducer",
    EEdigiCollection = cms.InputTag("ecalDigis","eeDigis"),
    EBdigiCollection = cms.InputTag("ecalDigis","ebDigis"),
    EEhitCollection = cms.string('EcalUncalibRecHitsEE'),
    beta = cms.double(1.7),
    AlphaBetaFilename = cms.untracked.string('NOFILE'),
    MinAmplEndcap = cms.double(8.0),
    MinAmplBarrel = cms.double(8.0),
    UseDynamicPedestal = cms.bool(True),
    alpha = cms.double(1.2),
    EBhitCollection = cms.string('EcalUncalibRecHitsEB')
)


