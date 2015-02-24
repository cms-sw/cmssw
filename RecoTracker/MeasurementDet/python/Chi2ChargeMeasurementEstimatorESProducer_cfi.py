import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripClusterizer.SiStripClusterChargeCut_cfi import *

Chi2ChargeMeasurementEstimator = cms.ESProducer("Chi2ChargeMeasurementEstimatorESProducer",
    ComponentName        = cms.string('Chi2Charge'),
    nSigma               = cms.double(3.0),
    MaxChi2              = cms.double(30.0),
    clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight')),
    pTChargeCutThreshold = cms.double(-1.)
)


