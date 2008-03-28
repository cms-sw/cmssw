import FWCore.ParameterSet.Config as cms

import copy
from TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi import *
Chi2MeasurementEstimatorForOutIn = copy.deepcopy(Chi2MeasurementEstimator)
Chi2MeasurementEstimatorForOutIn.ComponentName = 'Chi2ForOutIn'
Chi2MeasurementEstimatorForOutIn.MaxChi2 = 500.
Chi2MeasurementEstimatorForOutIn.nSigma = 3.

