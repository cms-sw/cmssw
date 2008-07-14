import FWCore.ParameterSet.Config as cms

# includes for tracking rechits
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
from RecoLocalTracker.SiPixelRecHits.PixelCPEParmError_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.StripCPE_cfi import *
from RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi import *
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
from RecoTracker.TkNavigation.NavigationSchoolESProducer_cff import *
from RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi import *
from TrackingTools.MaterialEffects.MaterialPropagator_cfi import *
# module to make track seeds
from RecoTracker.TkSeedGenerator.GlobalCombinedSeeds_cff import *
from RecoTracker.TkSeedGenerator.GlobalSeedsFromPairsWithVertices_cff import *
from RecoTracker.TkSeedGenerator.GlobalSeedsFromTripletsWithVertices_cff import *
newSeedFromPairs = RecoTracker.TkSeedGenerator.GlobalSeedsFromPairsWithVertices_cfi.globalSeedsFromPairsWithVertices.clone()
newSeedFromPairs.RegionFactoryPSet.RegionPSet.ptMin = 0.9  
newSeedFromTriplets = RecoTracker.TkSeedGenerator.GlobalSeedsFromTripletsWithVertices_cfi.globalSeedsFromTripletsWithVertices.clone()   
newCombinedSeeds = RecoTracker.TkSeedGenerator.GlobalCombinedSeeds_cfi.globalCombinedSeeds.clone()
newCombinedSeeds.PairCollection = 'newSeedFromPairs'
newCombinedSeeds.TripletCollection = 'newSeedFromTriplets' 
# seed module 
from RecoEgamma.EgammaElectronProducers.electronPixelSeeds_cfi import *

