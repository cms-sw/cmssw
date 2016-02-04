import FWCore.ParameterSet.Config as cms

from RecoTracker.TrackProducer.TrackRefitter_cfi import *
# Adjust to your needs:
#    replace TrackRefitter.src = "AlignmentTracks"
#    replace TrackRefitter.TrajectoryInEvent = true
# All conditions should be included centrally, but you'll need some for the lorentz angle:
#    include "CalibTracker/Configuration/data/SiStrip_FakeLorentzAngle.cff"
from TrackingTools.KalmanUpdators.KFUpdatorESProducer_cfi import *
from TrackingTools.TrackFitters.KFTrajectoryFitterESProducer_cfi import *
from TrackingTools.TrackFitters.KFTrajectorySmootherESProducer_cfi import *
from TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi import *
from TrackingTools.GeomPropagators.AnalyticalPropagator_cfi import *
from TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi import *
# Not to loose hits/tracks, we might want to open the allowed chi^2 contribution for single hits:
#    replace Chi2MeasurementEstimator.MaxChi2 = 50. # untested, default 30
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
from RecoLocalTracker.SiPixelRecHits.PixelCPEParmError_cfi import *
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
# Take care to avoid interferences with other things 
# using KFTrajectoryFitter and KFTrajectorySmoother...
TrackRefitter.Propagator = 'AnalyticalPropagator'
KFTrajectoryFitter.Propagator = 'AnalyticalPropagator'
KFTrajectorySmoother.Propagator = 'AnalyticalPropagator'

