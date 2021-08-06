import FWCore.ParameterSet.Config as cms

ecalUncalibRecHitPhase2 = cms.EDProducer('EcalUncalibRecHitPhase2WeightsProducer',
    EBhitCollection = cms.string('EcalUncalibRecHitsEB'),
    tRise = cms.double(0.2),
    tFall = cms.double(2.),
    BarrelDigis = cms.InputTag("simEcalUnsuppressedDigis",""),
    # weights calculated averaging the full EB
    weights = cms.vdouble( -0.121016, -0.119899, -0.120923, -0.0848959, 0.261041, 0.509881, 0.373591, 0.134899, -0.0233605, -0.0913195, -0.112452, -0.118596, -0,121737, -0,121737, -0,121737, -0,121737 )
)
