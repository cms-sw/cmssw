#############################
# This cfg configures the part of reconstruction 
# in FastSim to be done before event mixing
# FastSim mixes tracker information on reconstruction level,
# so tracks are recontructed before mixing.
# At present, only the generalTrack collection, produced with iterative tracking is mixed.
#############################

import FWCore.ParameterSet.Config as cms


# iterative tracking relies on the beamspot
from RecoVertex.BeamSpotProducer.BeamSpot_cff import offlineBeamSpot

# and of course it needs tracker hits
from FastSimulation.TrackingRecHitProducer.TrackingRecHitProducer_cfi import fastTrackerRecHits

from FastSimulation.TrackingRecHitProducer.FastTrackerRecHitMatcher_cfi import fastMatchedTrackerRecHits
import FastSimulation.Tracking.FastTrackerRecHitCombiner_cfi

fastMatchedTrackerRecHitCombinations = FastSimulation.Tracking.FastTrackerRecHitCombiner_cfi.fastTrackerRecHitCombinations.clone(
    simHit2RecHitMap = cms.InputTag("fastMatchedTrackerRecHits","simHit2RecHitMap")
    )

# confusing name for the file that imports 
# the fitters used by the TrackProducer
# 
from TrackingTools.MaterialEffects.Propagators_cff import *
from TrackingTools.TrackFitters.TrackFitters_cff import *
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilderWithoutRefit_cfi import *
from TrackingTools.KalmanUpdators.KFUpdatorESProducer_cfi import *
from TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi import *

#  MeasurementTrackerEvent
from RecoLocalTracker.SiPixelRecHits.PixelCPEGeneric_cfi import *
from RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cff import *
from FastSimulation.Tracking.MeasurementTrackerEventProducer_cfi import MeasurementTrackerEvent
# services needed by tracking
from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import TransientTrackBuilderESProducer
from RecoTracker.TkNavigation.NavigationSchoolESProducer_cfi import navigationSchoolESProducer

from FastSimulation.Tracking.iterativeTk_cff import *
from TrackingTools.TrackFitters.TrackFitters_cff import *
from RecoLocalTracker.SiPixelRecHits.SiPixelTemplateStoreESProducer_cfi import *

reconstruction_befmix = cms.Sequence(
    offlineBeamSpot
    * fastTrackerRecHits
    * fastMatchedTrackerRecHits
    * fastMatchedTrackerRecHitCombinations
    * MeasurementTrackerEvent
    * iterTracking
    )
