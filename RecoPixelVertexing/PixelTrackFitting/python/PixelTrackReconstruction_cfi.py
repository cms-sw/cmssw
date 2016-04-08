import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

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
        ComponentName = cms.string('PixelTrackCleanerBySharedHits')
    )
)

eras.trackingLowPU.toModify(PixelTrackReconstructionBlock,
    OrderedHitsFactoryPSet = dict(
        SeedingLayers = "PixelLayerTripletsPreSplitting",
        GeneratorPSet = dict(SeedComparitorPSet = dict(clusterShapeCacheSrc = "siPixelClusterShapeCachePreSplitting"))
    ),
)
