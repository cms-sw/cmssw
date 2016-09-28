import FWCore.ParameterSet.Config as cms

from RecoPixelVertexing.PixelTrackFitting.PixelFitterByHelixProjections_cfi import *
from RecoTracker.TkTrackingRegions.GlobalTrackingRegionFromBeamSpot_cfi import *
from RecoPixelVertexing.PixelTriplets.PixelTripletHLTGenerator_cfi import *


PixelTrackReconstructionBlock = cms.PSet (
    FitterPSet = cms.PSet(
        PixelFitterByHelixProjections
    ),
    useFilterWithES = cms.bool(False),
    FilterPSet = cms.PSet(
        nSigmaInvPtTolerance = cms.double(0.0),
        nSigmaTipMaxTolerance = cms.double(0.0),
        ComponentName = cms.string('PixelTrackFilterByKinematics'),
        chi2 = cms.double(1000.0),
        ptMin = cms.double(0.1),
        tipMax = cms.double(1.0)
    ),
    RegionFactoryPSet = cms.PSet(
        RegionPsetFomBeamSpotBlock,
        ComponentName = cms.string('GlobalRegionProducerFromBeamSpot')
    ),
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardHitTripletGenerator'),
        SeedingLayers = cms.InputTag('PixelLayerTriplets'),
        GeneratorPSet = cms.PSet(
            PixelTripletHLTGeneratorWithFilter
        )
    ),
    CleanerPSet = cms.PSet(
        ComponentName = cms.string('PixelTrackCleanerBySharedHits'),
        useQuadrupletAlgo = cms.bool(False),
    )
)

_OrderedHitsFactoryPSet_LowPU_Phase1PU70 = dict(
    SeedingLayers = "PixelLayerTripletsPreSplitting",
    GeneratorPSet = dict(SeedComparitorPSet = dict(clusterShapeCacheSrc = "siPixelClusterShapeCachePreSplitting"))
)
from Configuration.Eras.Modifier_trackingLowPU_cff import trackingLowPU
trackingLowPU.toModify(PixelTrackReconstructionBlock, OrderedHitsFactoryPSet = _OrderedHitsFactoryPSet_LowPU_Phase1PU70)
from Configuration.Eras.Modifier_trackingPhase1PU70_cff import trackingPhase1PU70
trackingPhase1PU70.toModify(PixelTrackReconstructionBlock,
    SeedMergerPSet = cms.PSet(
        layerList = cms.PSet(refToPSet_ = cms.string('PixelSeedMergerQuadruplets')),
        addRemainingTriplets = cms.bool(False),
        mergeTriplets = cms.bool(True),
        ttrhBuilderLabel = cms.string('PixelTTRHBuilderWithoutAngle')
    ),
    FilterPSet = dict(
        chi2 = 50.0,
        tipMax = 0.05
    ),
    RegionFactoryPSet = dict(RegionPSet = dict(originRadius =  0.02)),
    OrderedHitsFactoryPSet = _OrderedHitsFactoryPSet_LowPU_Phase1PU70,
)
from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
trackingPhase2PU140.toModify(PixelTrackReconstructionBlock,
    SeedMergerPSet = cms.PSet(
        layerList = cms.PSet(refToPSet_ = cms.string('PixelSeedMergerQuadruplets')),
        addRemainingTriplets = cms.bool(False),
        mergeTriplets = cms.bool(True),
        ttrhBuilderLabel = cms.string('PixelTTRHBuilderWithoutAngle')
    ),
    FilterPSet = dict(
        chi2 = 50.0,
        tipMax = 0.05
    ),
    RegionFactoryPSet = dict(RegionPSet = dict(originRadius =  0.02)),
    OrderedHitsFactoryPSet = dict(
      SeedingLayers = "PixelLayerTripletsPreSplitting",
      GeneratorPSet = dict(SeedComparitorPSet = dict(clusterShapeCacheSrc = "siPixelClusterShapeCachePreSplitting"),
                           maxElement = 0)
    )
)

