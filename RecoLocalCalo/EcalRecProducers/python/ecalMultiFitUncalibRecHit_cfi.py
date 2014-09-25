import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.EcalRecProducers.ecalPulseShapeParameters_cff import *

ecalMultiFitUncalibRecHit = cms.EDProducer("EcalUncalibRecHitProducer",
    EBdigiCollection = cms.InputTag("ecalDigis","ebDigis"),
    EEdigiCollection = cms.InputTag("ecalDigis","eeDigis"),
    EBhitCollection = cms.string("EcalUncalibRecHitsEB"),
    EEhitCollection = cms.string('EcalUncalibRecHitsEE'),

    # for multifit method
    EcalPulseShapeParameters = cms.PSet( ecal_pulse_shape_parameters ),
    activeBXs = cms.vint32(-5,-4,-3,-2,-1,0,1,2,3,4),
    ampErrorCalculation = cms.bool(True),

    # decide which algorithm to be use to calculate the jitter
    timealgo = cms.string("RatioMethod"),

    # for ratio method
    EBtimeFitParameters = cms.vdouble(-2.015452e+00, 3.130702e+00, -1.234730e+01, 4.188921e+01, -8.283944e+01, 9.101147e+01, -5.035761e+01, 1.105621e+01),
    EEtimeFitParameters = cms.vdouble(-2.390548e+00, 3.553628e+00, -1.762341e+01, 6.767538e+01, -1.332130e+02, 1.407432e+02, -7.541106e+01, 1.620277e+01),
    EBamplitudeFitParameters = cms.vdouble(1.138,1.652),
    EEamplitudeFitParameters = cms.vdouble(1.890,1.400),
    EBtimeFitLimits_Lower = cms.double(0.2),
    EBtimeFitLimits_Upper = cms.double(1.4),
    EEtimeFitLimits_Lower = cms.double(0.2),
    EEtimeFitLimits_Upper = cms.double(1.4),
    # for time error
    EBtimeConstantTerm= cms.double(.6),
    EEtimeConstantTerm= cms.double(1.0),
   
    ebPulseShape = cms.vdouble( 5.2e-05,-5.26e-05 , 6.66e-05, 0.1168, 0.7575, 1.,  0.8876, 0.6732, 0.4741,  0.3194 ),
    eePulseShape = cms.vdouble( 5.2e-05,-5.26e-05 , 6.66e-05, 0.1168, 0.7575, 1.,  0.8876, 0.6732, 0.4741,  0.3194 ),   

    # for kPoorReco flag
    kPoorRecoFlagEB = cms.bool(True),
    kPoorRecoFlagEE = cms.bool(False),
    chi2ThreshEB_ = cms.double(65.0),
    chi2ThreshEE_ = cms.double(50.0),
                                           
    algo = cms.string("EcalUncalibRecHitWorkerMultiFit")
)
