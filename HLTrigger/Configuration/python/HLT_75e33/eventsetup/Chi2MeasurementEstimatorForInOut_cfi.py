import FWCore.ParameterSet.Config as cms

Chi2MeasurementEstimatorForInOut = cms.ESProducer("Chi2MeasurementEstimatorESProducer",
    ComponentName = cms.string('Chi2ForInOut'),
    MaxChi2 = cms.double(100.0),
    MaxDisplacement = cms.double(100),
    MaxSagitta = cms.double(-1),
    MinPtForHitRecoveryInGluedDet = cms.double(1000000000000),
    MinimalTolerance = cms.double(0.5),
    appendToDataLabel = cms.string(''),
    nSigma = cms.double(3)
)
