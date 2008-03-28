import FWCore.ParameterSet.Config as cms

import copy
from TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi import *
chi2CutForConversionTrajectoryBuilder = copy.deepcopy(Chi2MeasurementEstimator)
chi2CutForConversionTrajectoryBuilder.ComponentName = 'eleLooseChi2'
chi2CutForConversionTrajectoryBuilder.MaxChi2 = 100000.
chi2CutForConversionTrajectoryBuilder.nSigma = 3.

