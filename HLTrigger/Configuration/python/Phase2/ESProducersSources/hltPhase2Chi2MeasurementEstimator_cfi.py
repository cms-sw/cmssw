import FWCore.ParameterSet.Config as cms

from TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorDefault_cfi import (
    Chi2MeasurementEstimatorDefault as _Chi2MeasurementEstimatorDefault,
)

hltPhase2Chi2MeasurementEstimator = _Chi2MeasurementEstimatorDefault.clone()
