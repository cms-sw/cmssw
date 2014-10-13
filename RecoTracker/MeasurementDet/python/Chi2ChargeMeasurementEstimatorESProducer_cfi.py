import FWCore.ParameterSet.Config as cms

Chi2ChargeMeasurementEstimator = cms.ESProducer("Chi2ChargeMeasurementEstimatorESProducer",
    ComponentName        = cms.string('Chi2Charge'),
    nSigma               = cms.double(3.0),
    MaxChi2              = cms.double(30.0),
    minGoodStripCharge   = cms.double(2069),
    pTChargeCutThreshold = cms.double(-1.)
)


