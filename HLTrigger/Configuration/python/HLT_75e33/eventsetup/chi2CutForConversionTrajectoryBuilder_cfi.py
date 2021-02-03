import FWCore.ParameterSet.Config as cms

chi2CutForConversionTrajectoryBuilder = cms.ESProducer("Chi2MeasurementEstimatorESProducer",
    ComponentName = cms.string('eleLooseChi2'),
    MaxChi2 = cms.double(100000.0),
    MaxDisplacement = cms.double(100.0),
    MaxSagitta = cms.double(-1),
    MinPtForHitRecoveryInGluedDet = cms.double(1000000000000),
    MinimalTolerance = cms.double(0.5),
    appendToDataLabel = cms.string(''),
    nSigma = cms.double(3)
)
