import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripClusterizer.SiStripClusterChargeCut_cfi import *

from RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimatorDefault_cfi import Chi2ChargeMeasurementEstimatorDefault
Chi2ChargeMeasurementEstimator = Chi2ChargeMeasurementEstimatorDefault.clone()
Chi2ChargeMeasurementEstimator.clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight'))


