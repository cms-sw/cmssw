import FWCore.ParameterSet.Config as cms

# Magntic field
# Geometry (all CMS)
# Tracker Geometry Builder
# Tracker Numbering Builder
# Reco geometry 
#from RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi import *
# for Transient rechits?
from RecoLocalTracker.SiPixelRecHits.PixelCPEParmError_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
#-ap   include "CalibTracker/Configuration/data/SiPixelLorentzAngle/SiPixelLorentzAngle_Fake.cff"
# include "RecoTracker/TransientTrackingRecHit/data/TransientTrackingRecHitBuilderWithoutRefit.cfi"
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
from RecoPixelVertexing.PixelTrackFitting.pixelTracks_cfi import pixelTracks
from RecoTracker.TkTrackingRegions.globalTrackingRegionFromBeamSpot_cfi import globalTrackingRegionFromBeamSpot as _globalTrackingRegionFromBeamSpot
from RecoTracker.TkHitPairs.hitPairEDProducer_cfi import hitPairEDProducer as _hitPairEDProducer
from RecoPixelVertexing.PixelTriplets.pixelTripletHLTEDProducer_cfi import pixelTripletHLTEDProducer as _pixelTripletHLTEDProducer
from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
import RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi
from RecoPixelVertexing.PixelTriplets.pixelQuadrupletMergerEDProducer_cfi import pixelQuadrupletMergerEDProducer as _pixelQuadrupletMergerEDProducer
from RecoPixelVertexing.PixelTriplets.quadrupletseedmerging_cff import *

from Configuration.Eras.Modifier_trackingLowPU_cff import trackingLowPU
from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140

# TrackingRegion
pixelTracksTrackingRegions = _globalTrackingRegionFromBeamSpot.clone()
trackingPhase2PU140.toModify(pixelTracksTrackingRegions, RegionPSet = dict(originRadius =  0.02))

# Hit ntuplets
pixelTracksHitDoublets = _hitPairEDProducer.clone(
    clusterCheck = "",
    seedingLayers = "PixelLayerTriplets",
    trackingRegions = "pixelTracksTrackingRegions",
    maxElement = 0,
    produceIntermediateHitDoublets = True,
)
_seedingLayers = dict(seedingLayers = "PixelLayerTripletsPreSplitting")
trackingLowPU.toModify(pixelTracksHitDoublets, **_seedingLayers)
trackingPhase2PU140.toModify(pixelTracksHitDoublets, **_seedingLayers)

pixelTracksHitTriplets = _pixelTripletHLTEDProducer.clone(
    doublets = "pixelTracksHitDoublets",
    produceSeedingHitSets = True,
    SeedComparitorPSet = RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi.LowPtClusterShapeSeedComparitor.clone()
)
_SeedComparitorPSet = dict(SeedComparitorPSet = dict(clusterShapeCacheSrc = "siPixelClusterShapeCachePreSplitting"))
trackingLowPU.toModify(pixelTracksHitTriplets, **_SeedComparitorPSet)
trackingPhase2PU140.toModify(pixelTracksHitTriplets, maxElement=0, **_SeedComparitorPSet)

pixelTracksHitQuadruplets = _pixelQuadrupletMergerEDProducer.clone(
    triplets = "pixelTracksHitTriplets",
    layerList = dict(refToPSet_ = cms.string("PixelSeedMergerQuadruplets")),
)

pixelTracksSequence = cms.Sequence(
    pixelTracksTrackingRegions +
    pixelTracksHitDoublets +
    pixelTracksHitTriplets +
    pixelFitterByHelixProjections +
    pixelTrackFilterByKinematics +
    pixelTracks
)
_pixelTracksSequence_quad = pixelTracksSequence.copy()
_pixelTracksSequence_quad.replace(pixelTracksHitTriplets, pixelTracksHitTriplets+pixelTracksHitQuadruplets)
trackingPhase2PU140.toReplaceWith(pixelTracksSequence, _pixelTracksSequence_quad)
