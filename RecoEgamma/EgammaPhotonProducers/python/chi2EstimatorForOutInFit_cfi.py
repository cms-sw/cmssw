import FWCore.ParameterSet.Config as cms

import TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi
Chi2MeasurementEstimatorForOutIn = TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi.Chi2MeasurementEstimator.clone(
    ComponentName   = 'Chi2ForOutIn',
    MaxChi2         = 500.,
    nSigma          = 3.,
    MaxDisplacement = 100,
    MaxSagitta      = -1. 
)
