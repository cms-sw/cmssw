import FWCore.ParameterSet.Config as cms

from RecoTracker.TkHitPairs.hitPairEDProducer_cfi import hitPairEDProducer as _hitPairEDProducer
from RecoPixelVertexing.PixelTriplets.pixelTripletHLTEDProducer_cfi import pixelTripletHLTEDProducer as _pixelTripletHLTEDProducer
from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
from RecoPixelVertexing.PixelLowPtUtilities.trackCleaner_cfi import *
from RecoPixelVertexing.PixelTrackFitting.pixelFitterByConformalMappingAndLine_cfi import *
from RecoHI.HiTracking.HIPixelTrackFilter_cff import *
from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *

# Hit ntuplets
hiConformalPixelTracksHitDoublets = _hitPairEDProducer.clone(
    clusterCheck = "",
    seedingLayers = "PixelLayerTriplets",
    trackingRegions = "hiTrackingRegionWithVertex",
    maxElement = 50000000,
    produceIntermediateHitDoublets = True,
)

hiConformalPixelTracksHitTriplets = _pixelTripletHLTEDProducer.clone(
    doublets = "hiConformalPixelTracksHitDoublets",
    maxElement = 5000000, # increase threshold for triplets in generation step (default: 100000)
    produceSeedingHitSets = True,
)

# Pixel tracks
hiConformalPixelTracks = cms.EDProducer("PixelTrackProducer",
                                        
                                        #passLabel  = cms.string('Pixel triplet low-pt tracks with vertex constraint'),
                                        
                                        # Ordered Hits
                                        SeedingHitSets = cms.InputTag("hiConformalPixelTracksHitTriplets"),
                                        
                                        # Fitter
                                        Fitter = cms.InputTag('pixelFitterByConformalMappingAndLine'),
                                        
                                        # Filter
                                        Filter = cms.InputTag("hiConformalPixelFilter"),
                                        
                                        # Cleaner
                                        Cleaner = cms.string("trackCleaner")
                                        )

hiConformalPixelTracksSequence = cms.Sequence(
    hiTrackingRegionWithVertex +
    hiConformalPixelTracksHitDoublets +
    hiConformalPixelTracksHitTriplets +
    pixelFitterByConformalMappingAndLine +
    hiConformalPixelFilter +
    hiConformalPixelTracks
)
