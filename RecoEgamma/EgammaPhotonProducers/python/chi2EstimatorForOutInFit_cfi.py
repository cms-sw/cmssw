import FWCore.ParameterSet.Config as cms

import TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi
Chi2MeasurementEstimatorForOutIn = TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi.Chi2MeasurementEstimator.clone()
Chi2MeasurementEstimatorForOutIn.ComponentName = 'Chi2ForOutIn'
Chi2MeasurementEstimatorForOutIn.MaxChi2 = 500.
Chi2MeasurementEstimatorForOutIn.nSigma = 3.
Chi2MeasurementEstimatorForOutIn.MaxDisplacement = 100
Chi2MeasurementEstimatorForOutIn.MaxSagitta = -1. 

