import FWCore.ParameterSet.Config as cms

from RecoTracker.TkHitPairs.hitPairEDProducer_cfi import hitPairEDProducer as _hitPairEDProducer
from RecoPixelVertexing.PixelTriplets.pixelTripletHLTEDProducer_cfi import pixelTripletHLTEDProducer as _pixelTripletHLTEDProducer
from RecoPixelVertexing.PixelTriplets.pixelQuadrupletEDProducer_cfi import pixelQuadrupletEDProducer as _pixelQuadrupletEDProducer
from RecoPixelVertexing.PixelTriplets.quadrupletseedmerging_cff import *
from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
from RecoPixelVertexing.PixelLowPtUtilities.trackCleaner_cfi import *
from RecoPixelVertexing.PixelTrackFitting.pixelFitterByHelixProjections_cfi import *
from RecoHI.HiTracking.HIPixelTrackFilter_cff import *
from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *
from RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi import *

#from RecoTracker.TkSeedingLayers.seedingLayersEDProducer_cfi import *

hiPixelLayerQuadruplets = PixelLayerTriplets.clone()
hiPixelLayerQuadruplets.layerList = PixelSeedMergerQuadruplets.layerList

# Hit ntuplets
hiPixel3PrimTracksHitDoublets = _hitPairEDProducer.clone(
    clusterCheck = "",
    seedingLayers = "PixelLayerTriplets",
    trackingRegions = "hiTrackingRegionWithVertex",
    maxElement = 0,
    produceIntermediateHitDoublets = True,
)
from Configuration.Eras.Modifier_trackingPhase1QuadProp_cff import trackingPhase1QuadProp
trackingPhase1QuadProp.toModify(hiPixel3PrimTracksHitDoublets,
    seedingLayers = "hiPixelLayerQuadruplets"
)


hiPixel3PrimTracksHitTriplets = _pixelTripletHLTEDProducer.clone(
    doublets = "hiPixel3PrimTracksHitDoublets",
    maxElement = 1000000, # increase threshold for triplets in generation step (default: 100000)
    produceSeedingHitSets = True,
    produceIntermediateHitTriplets = True,
)


# pixelQuadrupletMerger is not in use here. pp use it for trackingPhase1PU70
from RecoPixelVertexing.PixelTriplets.pixelQuadrupletMergerEDProducer_cfi import pixelQuadrupletMergerEDProducer as _pixelQuadrupletMergerEDProducer
from RecoPixelVertexing.PixelTriplets.quadrupletseedmerging_cff import *
_hiPixel3PrimTracksHitQuadrupletsMerging = _pixelQuadrupletMergerEDProducer.clone(
    triplets = "hiPixel3PrimTracksHitTriplets",
    layerList = dict(refToPSet_ = cms.string("PixelSeedMergerQuadruplets")),
)
####

hiPixel3PrimTracksHitQuadruplets = _pixelQuadrupletEDProducer.clone(
    triplets = "hiPixel3PrimTracksHitTriplets",
    extraHitRZtolerance = hiPixel3PrimTracksHitTriplets.extraHitRZtolerance,
    extraHitRPhitolerance = hiPixel3PrimTracksHitTriplets.extraHitRPhitolerance,
    maxChi2 = dict(
        pt1    = 0.8, pt2    = 2,
        value1 = 200, value2 = 100,
        enabled = True,
    ),
    extraPhiTolerance = dict(
        pt1    = 0.6, pt2    = 1,
        value1 = 0.15, value2 = 0.1,
        enabled = True,
    ),
    useBendingCorrection = True,
    fitFastCircle = True,
    fitFastCircleChi2Cut = True,
    SeedComparitorPSet = hiPixel3PrimTracksHitTriplets.SeedComparitorPSet
)

from RecoPixelVertexing.PixelTriplets.caHitQuadrupletEDProducer_cfi import caHitQuadrupletEDProducer as _caHitQuadrupletEDProducer
hiPixel3PrimTracksHitDoubletsCA = hiPixel3PrimTracksHitDoublets.clone()
hiPixel3PrimTracksHitDoubletsCA.layerPairs = [0,1,2]

hiPixel3PrimTracksHitQuadrupletsCA = _caHitQuadrupletEDProducer.clone(
    doublets = "hiPixel3PrimTracksHitDoubletsCA",
    extraHitRPhitolerance = hiPixel3PrimTracksHitTriplets.extraHitRPhitolerance,
    SeedComparitorPSet = hiPixel3PrimTracksHitTriplets.SeedComparitorPSet,
    maxChi2 = dict(
        pt1    = 0.7, pt2    = 2,
        value1 = 200, value2 = 50,
    ),
    useBendingCorrection = True,
    fitFastCircle = True,
    fitFastCircleChi2Cut = True,
    CAThetaCut = 0.0012,
    CAPhiCut = 0.2,
) 

# Pixel tracks
hiPixel3PrimTracks = cms.EDProducer("PixelTrackProducer",

    passLabel  = cms.string('Pixel triplet primary tracks with vertex constraint'),

    # Ordered Hits
    SeedingHitSets = cms.InputTag("hiPixel3PrimTracksHitTriplets"),
    #SeedingHitSets = cms.InputTag("hiPixel3PrimTracksHitQuadruplets"),
	
    # Fitter
    Fitter = cms.InputTag("pixelFitterByHelixProjections"),
	
    # Filter
    Filter = cms.InputTag("hiFilter"),
	
    # Cleaner
    Cleaner = cms.string("trackCleaner")
)
trackingPhase1QuadProp.toModify(hiPixel3PrimTracks,
    SeedingHitSets = cms.InputTag("hiPixel3PrimTracksHitQuadrupletsCA"),
)

hiPixel3PrimTracksSequence = cms.Sequence(
    hiTrackingRegionWithVertex +
    hiPixel3PrimTracksHitDoublets +
    hiPixel3PrimTracksHitTriplets +
    pixelFitterByHelixProjections +
    hiFilter +
    hiPixel3PrimTracks
)

#phase 1 changes
hiPixel3PrimTracksSequence_Phase1 = hiPixel3PrimTracksSequence.copy()
hiPixel3PrimTracksSequence_Phase1.replace(hiPixel3PrimTracksHitDoublets,hiPixelLayerQuadruplets+hiPixel3PrimTracksHitDoubletsCA)#can remove 'CA' to get regular seeds
hiPixel3PrimTracksSequence_Phase1.replace(hiPixel3PrimTracksHitTriplets,hiPixel3PrimTracksHitTriplets+hiPixel3PrimTracksHitQuadrupletsCA)#can remove 'CA' to get regular seeds
trackingPhase1QuadProp.toReplaceWith(hiPixel3PrimTracksSequence,hiPixel3PrimTracksSequence_Phase1)
