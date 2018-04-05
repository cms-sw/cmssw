import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaPhotonProducers.KFTrajectoryFitterForOutIn_cfi import *
from RecoEgamma.EgammaPhotonProducers.KFTrajectorySmootherForOutIn_cfi import *
import TrackingTools.TrackFitters.KFFittingSmoother_cfi
# KFFittingSmootherESProducer
KFFittingSmootherForOutIn = TrackingTools.TrackFitters.KFFittingSmoother_cfi.KFFittingSmoother.clone()
KFFittingSmootherForOutIn.ComponentName = 'KFFittingSmootherForOutIn'
KFFittingSmootherForOutIn.Fitter = 'KFFitterForOutIn'
KFFittingSmootherForOutIn.Smoother = 'KFSmootherForOutIn'
KFFittingSmootherForOutIn.EstimateCut = -1
KFFittingSmootherForOutIn.MinNumberOfHits = 3

