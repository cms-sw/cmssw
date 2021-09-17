import FWCore.ParameterSet.Config as cms

import TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi
Chi2MeasurementEstimatorForInOut = TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi.Chi2MeasurementEstimator.clone(
    ComponentName   = 'Chi2ForInOut',
    MaxChi2         = 100.,
    nSigma          = 3.,
    MaxDisplacement = 100,
    MaxSagitta      = -1
)
