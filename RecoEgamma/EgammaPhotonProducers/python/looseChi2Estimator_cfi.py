import FWCore.ParameterSet.Config as cms

import TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi
chi2CutForConversionTrajectoryBuilder = TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi.Chi2MeasurementEstimator.clone(
    ComponentName   = 'eleLooseChi2',
    MaxChi2         = 100000.,
    nSigma          = 3.,
    MaxDisplacement = 100.,
    MaxSagitta      = -1
)
