import FWCore.ParameterSet.Config as cms

muonSeededMeasurementEstimatorForOutIn = cms.ESProducer("Chi2MeasurementEstimatorESProducer",
    ComponentName = cms.string('muonSeededMeasurementEstimatorForOutIn'),
    MaxChi2 = cms.double(30.0),
    MaxDisplacement = cms.double(0.5),
    MaxSagitta = cms.double(-1.0),
    MinPtForHitRecoveryInGluedDet = cms.double(1000000000000),
    MinimalTolerance = cms.double(0.5),
    appendToDataLabel = cms.string(''),
    nSigma = cms.double(3.0)
)
