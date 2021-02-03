import FWCore.ParameterSet.Config as cms

muonSeededMeasurementEstimatorForInOut = cms.ESProducer("Chi2MeasurementEstimatorESProducer",
    ComponentName = cms.string('muonSeededMeasurementEstimatorForInOut'),
    MaxChi2 = cms.double(400.0),
    MaxDisplacement = cms.double(0.5),
    MaxSagitta = cms.double(2),
    MinPtForHitRecoveryInGluedDet = cms.double(1000000000000),
    MinimalTolerance = cms.double(0.5),
    appendToDataLabel = cms.string(''),
    nSigma = cms.double(4.0)
)
