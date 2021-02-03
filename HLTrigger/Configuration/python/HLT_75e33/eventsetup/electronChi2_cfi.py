import FWCore.ParameterSet.Config as cms

electronChi2 = cms.ESProducer("Chi2MeasurementEstimatorESProducer",
    ComponentName = cms.string('electronChi2'),
    MaxChi2 = cms.double(100.0),
    MaxDisplacement = cms.double(0.5),
    MaxSagitta = cms.double(2),
    MinPtForHitRecoveryInGluedDet = cms.double(1000000000000),
    MinimalTolerance = cms.double(0.5),
    appendToDataLabel = cms.string(''),
    nSigma = cms.double(3)
)
