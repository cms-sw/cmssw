import FWCore.ParameterSet.Config as cms

ecalRatioUncalibRecHit = cms.EDProducer("EcalUncalibRecHitProducer",
    EBdigiCollection = cms.InputTag("ecalDigis","ebDigis"),
    EEdigiCollection = cms.InputTag("ecalDigis","eeDigis"),
    EBhitCollection = cms.string("EcalUncalibRecHitsEB"),
    EEhitCollection = cms.string('EcalUncalibRecHitsEE'),
    algo = cms.string("EcalUncalibRecHitWorkerRatio"),
    algoPSet = cms.PSet(
      EBtimeFitParameters = cms.vdouble(-2.015452e+00, 3.130702e+00, -1.234730e+01, 4.188921e+01, -8.283944e+01, 9.101147e+01, -5.035761e+01, 1.105621e+01),
      EEtimeFitParameters = cms.vdouble(-2.390548e+00, 3.553628e+00, -1.762341e+01, 6.767538e+01, -1.332130e+02, 1.407432e+02, -7.541106e+01, 1.620277e+01),
      EBamplitudeFitParameters = cms.vdouble(1.138,1.652),
      EEamplitudeFitParameters = cms.vdouble(1.890,1.400),
      EBtimeFitLimits_Lower = cms.double(0.2),
      EBtimeFitLimits_Upper = cms.double(1.4),
      EEtimeFitLimits_Lower = cms.double(0.2),
      EEtimeFitLimits_Upper = cms.double(1.4),
      EBtimeConstantTerm = cms.double(.26),
      EEtimeConstantTerm = cms.double(.18),
    )
)
