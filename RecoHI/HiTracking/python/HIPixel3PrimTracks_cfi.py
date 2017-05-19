import FWCore.ParameterSet.Config as cms

from RecoTracker.TkHitPairs.hitPairEDProducer_cfi import hitPairEDProducer as _hitPairEDProducer
from RecoPixelVertexing.PixelTriplets.pixelTripletHLTEDProducer_cfi import pixelTripletHLTEDProducer as _pixelTripletHLTEDProducer
from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
from RecoPixelVertexing.PixelLowPtUtilities.trackCleaner_cfi import *
from RecoPixelVertexing.PixelTrackFitting.pixelFitterByHelixProjections_cfi import *
from RecoHI.HiTracking.HIPixelTrackFilter_cff import *
from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *
from RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi import *

# Hit ntuplets
hiPixel3PrimTracksHitDoublets = _hitPairEDProducer.clone(
    clusterCheck = "",
    seedingLayers = "PixelLayerTriplets",
    trackingRegions = "hiTrackingRegionWithVertex",
    maxElement = 0,
    produceIntermediateHitDoublets = True,
)

hiPixel3PrimTracksHitTriplets = _pixelTripletHLTEDProducer.clone(
    doublets = "hiPixel3PrimTracksHitDoublets",
    maxElement = 1000000, # increase threshold for triplets in generation step (default: 100000)
    produceSeedingHitSets = True,
)

# Pixel tracks
hiPixel3PrimTracks = cms.EDProducer("PixelTrackProducer",

    passLabel  = cms.string('Pixel triplet primary tracks with vertex constraint'),

    # Ordered Hits
    SeedingHitSets = cms.InputTag("hiPixel3PrimTracksHitTriplets"),
	
    # Fitter
    Fitter = cms.InputTag("pixelFitterByHelixProjections"),
	
    # Filter
    Filter = cms.InputTag("hiFilter"),
	
    # Cleaner
    Cleaner = cms.string("trackCleaner")
)

hiPixel3PrimTracksSequence = cms.Sequence(
    hiTrackingRegionWithVertex +
    hiPixel3PrimTracksHitDoublets +
    hiPixel3PrimTracksHitTriplets +
    pixelFitterByHelixProjections +
    hiFilter +
    hiPixel3PrimTracks
)
