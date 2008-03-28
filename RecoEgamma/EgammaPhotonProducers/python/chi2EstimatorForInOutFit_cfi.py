import FWCore.ParameterSet.Config as cms

import copy
from TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi import *
Chi2MeasurementEstimatorForInOut = copy.deepcopy(Chi2MeasurementEstimator)
Chi2MeasurementEstimatorForInOut.ComponentName = 'Chi2ForInOut'
Chi2MeasurementEstimatorForInOut.MaxChi2 = 100.
Chi2MeasurementEstimatorForInOut.nSigma = 3.

