import FWCore.ParameterSet.Config as cms

# producer of rechits starting from digis
ecalFixedAlphaBetaFitUncalibRecHit = cms.EDProducer("EcalUncalibRecHitProducer",
    EEdigiCollection = cms.InputTag("ecalDigis","eeDigis"),
    EBdigiCollection = cms.InputTag("ecalDigis","ebDigis"),
    EEhitCollection = cms.string("EcalUncalibRecHitsEE"),
    EBhitCollection = cms.string("EcalUncalibRecHitsEB"),
    algo = cms.string("EcalUncalibRecHitWorkerFixedAlphaBetaFit"), 
    algoPSet = cms.PSet(
      betaEB = cms.double(1.655),
      betaEE = cms.double(1.400),
      AlphaBetaFilename = cms.untracked.string("NOFILE"),
      MinAmplEndcap = cms.double(14.0),
      MinAmplBarrel = cms.double(8.0),
      UseDynamicPedestal = cms.bool(True),
      alphaEB = cms.double(1.138),
      alphaEE = cms.double(1.890)
    )
)
