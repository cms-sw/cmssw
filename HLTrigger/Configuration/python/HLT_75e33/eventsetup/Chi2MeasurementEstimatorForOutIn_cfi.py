import FWCore.ParameterSet.Config as cms

Chi2MeasurementEstimatorForOutIn = cms.ESProducer("Chi2MeasurementEstimatorESProducer",
    ComponentName = cms.string('Chi2ForOutIn'),
    MaxChi2 = cms.double(500.0),
    MaxDisplacement = cms.double(100),
    MaxSagitta = cms.double(-1.0),
    MinPtForHitRecoveryInGluedDet = cms.double(1000000000000),
    MinimalTolerance = cms.double(0.5),
    appendToDataLabel = cms.string(''),
    nSigma = cms.double(3)
)
