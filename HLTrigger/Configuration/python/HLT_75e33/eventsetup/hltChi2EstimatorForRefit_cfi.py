import FWCore.ParameterSet.Config as cms

hltChi2EstimatorForRefit = cms.ESProducer('Chi2MeasurementEstimatorESProducer',
    ComponentName = cms.string('hltChi2EstimatorForRefit'),
    MaxChi2 = cms.double(100000.0),
    nSigma = cms.double(3),
    MaxDisplacement = cms.double(0.5),
    MaxSagitta = cms.double(2),
    MinimalTolerance = cms.double(0.5),
    MinPtForHitRecoveryInGluedDet = cms.double(1000000000000),
    appendToDataLabel = cms.string('')
)
