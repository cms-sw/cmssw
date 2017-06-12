import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiPixelRecHits.PixelCPEParmError_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
import RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi
myTTRHBuilderWithoutAngle = RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi.ttrhbwr.clone(
    StripCPE = 'Fake',
    ComponentName = 'PixelTTRHBuilderWithoutAngle'
)
from RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi import *
from RecoTracker.TkSeedingLayers.TTRHBuilderWithoutAngle4PixelTriplets_cfi import *
from RecoPixelVertexing.PixelTrackFitting.pixelFitterByHelixProjections_cfi import pixelFitterByHelixProjections
from RecoPixelVertexing.PixelTrackFitting.pixelTrackFilterByKinematics_cfi import pixelTrackFilterByKinematics
from RecoPixelVertexing.PixelTrackFitting.pixelTrackCleanerBySharedHits_cfi import pixelTrackCleanerBySharedHits
from RecoPixelVertexing.PixelTrackFitting.pixelTracks_cfi import pixelTracks as _pixelTracks
from RecoTracker.TkTrackingRegions.globalTrackingRegion_cfi import globalTrackingRegion as _globalTrackingRegion
from RecoTracker.TkHitPairs.hitPairEDProducer_cfi import hitPairEDProducer as _hitPairEDProducer
from RecoPixelVertexing.PixelTriplets.pixelTripletHLTEDProducer_cfi import pixelTripletHLTEDProducer as _pixelTripletHLTEDProducer
from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
import RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi
from RecoTracker.FinalTrackSelectors.trackAlgoPriorityOrder_cfi import trackAlgoPriorityOrder

# SEEDING LAYERS
import RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi
import RecoPixelVertexing.PixelTriplets.quadrupletseedmerging_cff
from RecoTracker.IterativeTracking.InitialStep_cff import initialStepSeedLayers, initialStepHitDoublets, initialStepHitQuadruplets
from RecoTracker.IterativeTracking.HighPtTripletStep_cff import highPtTripletStepClusters, highPtTripletStepSeedLayers, highPtTripletStepHitDoublets, highPtTripletStepHitTriplets

# TrackingRegion
pixelTracksTrackingRegions = _globalTrackingRegion.clone()


# Pixel Quadruplets Tracking
pixelTracksQuadSeedLayers = initialStepSeedLayers.clone(
    BPix = dict(HitProducer = "siPixelRecHitsPreSplitting"),
    FPix = dict(HitProducer = "siPixelRecHitsPreSplitting")
)

pixelTracksQuadHitDoublets = initialStepHitDoublets.clone(
    clusterCheck = "",
    seedingLayers = "pixelTracksQuadSeedLayers",
    trackingRegions = "pixelTracksTrackingRegions"
)

pixelTracksQuadHitQuadruplets = initialStepHitQuadruplets.clone(
    doublets = "pixelTracksQuadHitDoublets",
    SeedComparitorPSet = dict(clusterShapeCacheSrc = 'siPixelClusterShapeCachePreSplitting')
)

pixelTracksQuad = _pixelTracks.clone(
    SeedingHitSets = "pixelTracksQuadHitQuadruplets"
)

pixelTracksQuadSequence = cms.Sequence(
    pixelTracksQuadSeedLayers +
    pixelTracksQuadHitDoublets +
    pixelTracksQuadHitQuadruplets +
    pixelTracksQuad
)

# Pixel Triplets Tracking
pixelTracksTripletSeedLayers = highPtTripletStepSeedLayers.clone(
    BPix = dict(skipClusters = cms.InputTag('pixelTracksTripletClusters'), HitProducer = "siPixelRecHitsPreSplitting"),
    FPix = dict(skipClusters = cms.InputTag('pixelTracksTripletClusters'), HitProducer = "siPixelRecHitsPreSplitting")
)

pixelTracksTripletClusters = highPtTripletStepClusters.clone(
    trackClassifier = cms.InputTag( '','QualityMasks' ),
    maxChi2 = cms.double( 3000.0 ),
    trajectories = cms.InputTag( "pixelTracksQuad" ),
    oldClusterRemovalInfo = cms.InputTag( "" ),
    pixelClusters = cms.InputTag( "siPixelClustersPreSplitting" ),
)

pixelTracksTripletHitDoublets = highPtTripletStepHitDoublets.clone(
    clusterCheck = "",
    seedingLayers = "pixelTracksTripletSeedLayers",
    trackingRegions = "pixelTracksTrackingRegions"
)

pixelTracksTripletHitTriplets = highPtTripletStepHitTriplets.clone(
    doublets = "pixelTracksTripletHitDoublets",
    SeedComparitorPSet = dict(clusterShapeCacheSrc = 'siPixelClusterShapeCachePreSplitting')
)

pixelTracksTriplet = _pixelTracks.clone(
    SeedingHitSets = "pixelTracksTripletHitTriplets"
)

pixelTracksTripletSequence = cms.Sequence(
    pixelTracksTripletClusters +
    pixelTracksTripletSeedLayers +
    pixelTracksTripletHitDoublets +
    pixelTracksTripletHitTriplets +
    pixelTracksTriplet
)



pixelTracks = cms.EDProducer( "TrackListMerger",
                              ShareFrac = cms.double( 0.19 ),
                              writeOnlyTrkQuals = cms.bool( False ),
                              MinPT = cms.double( 0.05 ),
                              allowFirstHitShare = cms.bool( True ),
                              copyExtras = cms.untracked.bool( True ),
                              Epsilon = cms.double( -0.001 ),
                              selectedTrackQuals = cms.VInputTag( 'pixelTracksTriplet','pixelTracksQuad' ),
                              indivShareFrac = cms.vdouble( 1.0, 1.0 ),
                              MaxNormalizedChisq = cms.double( 1000.0 ),
                              copyMVA = cms.bool( False ),
                              FoundHitBonus = cms.double( 5.0 ),
                              setsToMerge = cms.VPSet(
                                  cms.PSet(  pQual = cms.bool( False ),
                                             tLists = cms.vint32( 0, 1 )
                                  )
                              ),
                              MinFound = cms.int32( 3 ),
                              hasSelector = cms.vint32( 0, 0 ),
                              TrackProducers = cms.VInputTag( 'pixelTracksTriplet','pixelTracksQuad' ),
                              LostHitPenalty = cms.double( 20.0 ),
                              newQuality = cms.string( "confirmed" ),
                              trackAlgoPriorityOrder = cms.string("trackAlgoPriorityOrder")
)

pixelTracksSequence = cms.Sequence(
    pixelTracksTrackingRegions +
    pixelFitterByHelixProjections +
    pixelTrackFilterByKinematics +
    pixelTracksQuadSequence +
    pixelTracksTripletSequence +
    pixelTracks
)
