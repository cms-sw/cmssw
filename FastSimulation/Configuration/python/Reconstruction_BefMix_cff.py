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
from FastSimulation.TrackingRecHitProducer.SiTrackerGaussianSmearingRecHitConverter_cfi import siTrackerGaussianSmearingRecHits
from FastSimulation.TrackingRecHitProducer.FastTrackerRecHitMatcher_cfi import fastMatchedTrackerRecHits
import FastSimulation.Tracking.FastTrackerRecHitCombiner_cfi

fastMatchedTrackerRecHitCombinations = FastSimulation.Tracking.FastTrackerRecHitCombiner_cfi.fastTrackerRecHitCombinations.clone(
    simHit2RecHitMap = cms.InputTag("fastMatchedTrackerRecHits","simHit2RecHitMap")
    )

# FastSim stores the IdealMagneticFieldRecord and the TrackerInteractionGeometryRecord in a particular structure
# This extra layer is probably more confusing than it is useful and we should consider to remove it
from FastSimulation.ParticlePropagator.MagneticFieldMapESProducer_cfi import *

# confusing name for the file that imports 
# the fitters used by the TrackProducer
# 
from FastSimulation.Tracking.GSTrackFinalFitCommon_cff import *

#  MeasurementTrackerEvent
from RecoLocalTracker.SiPixelRecHits.PixelCPEGeneric_cfi import *
from RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cff import *
from FastSimulation.Tracking.MeasurementTrackerEventProducer_cfi import MeasurementTrackerEvent

# services needed by tracking
from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import TransientTrackBuilderESProducer
from RecoTracker.TkNavigation.NavigationSchoolESProducer_cfi import navigationSchoolESProducer


from FastSimulation.Tracking.iterativeTk_cff import *
from TrackingTools.TrackFitters.TrackFitters_cff import *

reconstruction_befmix = cms.Sequence(
    offlineBeamSpot
    * siTrackerGaussianSmearingRecHits
    * fastMatchedTrackerRecHits
    * fastMatchedTrackerRecHitCombinations
    * MeasurementTrackerEvent
    * iterTracking
    )
