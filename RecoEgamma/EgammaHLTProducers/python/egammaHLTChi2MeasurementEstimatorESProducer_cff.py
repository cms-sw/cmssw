import FWCore.ParameterSet.Config as cms

import copy
from TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi import *
egammaHLTChi2MeasurementEstimatorESProducer = copy.deepcopy(Chi2MeasurementEstimator)
egammaHLTChi2MeasurementEstimatorESProducer.ComponentName = 'egammaHLTChi2'
egammaHLTChi2MeasurementEstimatorESProducer.MaxChi2 = 5.
egammaHLTChi2MeasurementEstimatorESProducer.nSigma = 3.

