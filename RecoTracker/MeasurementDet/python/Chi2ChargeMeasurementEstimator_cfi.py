import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Modifier_tracker_apv_vfp30_2016_cff import tracker_apv_vfp30_2016 as _tracker_apv_vfp30_2016

from RecoLocalTracker.SiStripClusterizer.SiStripClusterChargeCut_cfi import *

from RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimatorDefault_cfi import Chi2ChargeMeasurementEstimatorDefault
Chi2ChargeMeasurementEstimator = Chi2ChargeMeasurementEstimatorDefault.clone(
    clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight'))
)
_tracker_apv_vfp30_2016.toModify(Chi2ChargeMeasurementEstimator, MinPtForHitRecoveryInGluedDet=0.9)


