import FWCore.ParameterSet.Config as cms

ecalUncalibRecHitPhase2 = cms.EDProducer('EcalUncalibRecHitPhase2WeightsProducer',
    EBhitCollection = cms.string('EcalUncalibRecHitsEB'),
    tRise = cms.double(0.2),
    tFall = cms.double(2.),
    BarrelDigis = cms.InputTag("simEcalUnsuppressedDigis","")
)