import FWCore.ParameterSet.Config as cms

# producer of rechits starting from digis
ecalFixedAlphaBetaFitUncalibRecHit = cms.EDProducer("EcalFixedAlphaBetaFitUncalibRecHitProducer",
    EEdigiCollection = cms.InputTag("ecalDigis","eeDigis"),
    EBdigiCollection = cms.InputTag("ecalDigis","ebDigis"),
    EEhitCollection = cms.string('EcalUncalibRecHitsEE'),
    betaEB = cms.double(1.7),
    betaEE = cms.double(1.37),
    AlphaBetaFilename = cms.untracked.string('NOFILE'),
    MinAmplEndcap = cms.double(14.0),
    MinAmplBarrel = cms.double(8.0),
    UseDynamicPedestal = cms.bool(True),
    alphaEB = cms.double(1.2),
    alphaEE = cms.double(1.63),
    EBhitCollection = cms.string('EcalUncalibRecHitsEB')
)


