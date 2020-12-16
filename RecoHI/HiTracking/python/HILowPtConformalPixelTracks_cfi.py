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
    clusterCheck    = "",
    seedingLayers   = "PixelLayerTriplets",
    trackingRegions = "hiTrackingRegionWithVertex",
    maxElement      = 50000000,
    produceIntermediateHitDoublets = True,
)

hiConformalPixelTracksHitTriplets = _pixelTripletHLTEDProducer.clone(
    doublets   = "hiConformalPixelTracksHitDoublets",
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





###Pixel Tracking -  PhaseI geometry

#Tracking regions - use PV from pp tracking
from RecoTracker.TkTrackingRegions.globalTrackingRegionWithVertices_cfi import globalTrackingRegionWithVertices
hiConformalPixelTracksPhase1TrackingRegions = globalTrackingRegionWithVertices.clone(
    RegionPSet = dict(
	precise = True,
	useMultipleScattering = False,
	useFakeVertices  = False,
	beamSpot         = "offlineBeamSpot",
	useFixedError    = True,
	nSigmaZ          = 3.0,
	sigmaZVertex     = 3.0,
	fixedError       = 0.2,
	VertexCollection = "offlinePrimaryVertices",
	ptMin            = 0.3,
	useFoundVertices = True,
	originRadius     = 0.2 
    )
)

# SEEDING LAYERS
# Using 4 layers layerlist
from RecoTracker.IterativeTracking.LowPtQuadStep_cff import lowPtQuadStepSeedLayers
hiConformalPixelTracksPhase1SeedLayers = lowPtQuadStepSeedLayers.clone(
    BPix = cms.PSet( 
	HitProducer = cms.string('siPixelRecHits'),
        TTRHBuilder = cms.string('WithTrackAngle'),
    ),
    FPix = cms.PSet( 
        HitProducer = cms.string('siPixelRecHits'),
        TTRHBuilder = cms.string('WithTrackAngle'),
    )
)


# Hit ntuplets
from RecoTracker.IterativeTracking.LowPtQuadStep_cff import lowPtQuadStepHitDoublets
hiConformalPixelTracksPhase1HitDoubletsCA = lowPtQuadStepHitDoublets.clone(
    seedingLayers   = "hiConformalPixelTracksPhase1SeedLayers",
    trackingRegions = "hiConformalPixelTracksPhase1TrackingRegions"
)


from RecoTracker.IterativeTracking.LowPtQuadStep_cff import lowPtQuadStepHitQuadruplets
hiConformalPixelTracksPhase1HitQuadrupletsCA = lowPtQuadStepHitQuadruplets.clone(
    doublets   = "hiConformalPixelTracksPhase1HitDoubletsCA",
    CAPhiCut   = 0.2,
    CAThetaCut = 0.0012,
    SeedComparitorPSet = dict( 
       ComponentName = 'none'
    ),
    extraHitRPhitolerance = 0.032,
    maxChi2 = dict(
       enabled = True,
       pt1     = 0.7,
       pt2     = 2,
       value1  = 200,
       value2  = 50
    )
)

#Filter
hiConformalPixelTracksPhase1Filter = hiConformalPixelFilter.clone(
    VertexCollection = "offlinePrimaryVertices",
    chi2   = 999.9,
    lipMax = 999.0,
    nSigmaLipMaxTolerance = 999.9,
    nSigmaTipMaxTolerance = 999.0,
    ptMax  = 999999,
    ptMin  = 0.30,
    tipMax = 999.0
)

from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify(hiConformalPixelTracks,
    Cleaner = 'pixelTrackCleanerBySharedHits',
    Filter  = "hiConformalPixelTracksPhase1Filter",
    Fitter  = "pixelFitterByConformalMappingAndLine",
    SeedingHitSets = "hiConformalPixelTracksPhase1HitQuadrupletsCA",
)

hiConformalPixelTracksTask = cms.Task(
    hiTrackingRegionWithVertex ,
    hiConformalPixelTracksHitDoublets ,
    hiConformalPixelTracksHitTriplets ,
    pixelFitterByConformalMappingAndLine ,
    hiConformalPixelFilter ,
    hiConformalPixelTracks
)

hiConformalPixelTracksTaskPhase1 = cms.Task(
    hiConformalPixelTracksPhase1TrackingRegions ,
    hiConformalPixelTracksPhase1SeedLayers ,
    hiConformalPixelTracksPhase1HitDoubletsCA ,
    hiConformalPixelTracksPhase1HitQuadrupletsCA ,
    pixelFitterByConformalMappingAndLine ,
    hiConformalPixelTracksPhase1Filter ,
    hiConformalPixelTracks
)
hiConformalPixelTracksSequencePhase1 = cms.Sequence(hiConformalPixelTracksTaskPhase1)
