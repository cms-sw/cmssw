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
#from FastSimulation.TrackingRecHitProducer.SiTrackerGaussianSmearingRecHitConverter_cfi import siTrackerGaussianSmearingRecHits
from FastSimulation.TrackingRecHitProducer.NewRecHitConverter_NoMerge_cfi import trackingRecHitProducerNoMerge as siTrackerGaussianSmearingRecHits
#from FastSimulation.TrackingRecHitProducer.NewRecHitConverter_Example_cfi import trackingRecHitProducer as siTrackerGaussianSmearingRecHits
#from FastSimulation.TrackingRecHitProducer.NewRecHitConverter_Example_cfi import trackingRecHitProducer_alt as siTrackerGaussianSmearingRecHits

resolutionsPIXELS = cms.EDProducer(
    "TrackerHitResolution",
    recHitsMap = cms.InputTag("siTrackerGaussianSmearingRecHits","simHit2RecHitMap"),
    simHits = cms.InputTag("famosSimHits","TrackerHits"),
    select = cms.string("subdetId==BPX || subdetId==FPX"),
    
    nbinsx=cms.int32(100),nbinsy=cms.int32(100),
    
    xposmin=cms.double(-0.01),xposmax=cms.double(0.01),
    yposmin=cms.double(-0.01),yposmax=cms.double(0.01),
    
    xerrmin=cms.double(0.0001),xerrmax=cms.double(0.01),
    yerrmin=cms.double(0.0001),yerrmax=cms.double(0.01)
)

resolutionsSTRIPS = cms.EDProducer(
    "TrackerHitResolution",
    recHitsMap = cms.InputTag("siTrackerGaussianSmearingRecHits","simHit2RecHitMap"),
    simHits = cms.InputTag("famosSimHits","TrackerHits"),
    select = cms.string("subdetId==TIB || subdetId==TID || subdetId==TOB || subdetId==TEC"),
    
    nbinsx=cms.int32(50),nbinsy=cms.int32(50),
    
    xposmin=cms.double(-0.02),xposmax=cms.double(0.02),
    yposmin=cms.double(-12),yposmax=cms.double(12),
    
    xerrmin=cms.double(0.001),xerrmax=cms.double(0.01),
    yerrmin=cms.double(1),yerrmax=cms.double(7)
)


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

reconstruction_befmix = cms.Sequence(
    offlineBeamSpot
    * siTrackerGaussianSmearingRecHits
    
    * resolutionsPIXELS
    * resolutionsSTRIPS
    
    * fastMatchedTrackerRecHits
    * fastMatchedTrackerRecHitCombinations
    * MeasurementTrackerEvent
    * iterTracking
    )
