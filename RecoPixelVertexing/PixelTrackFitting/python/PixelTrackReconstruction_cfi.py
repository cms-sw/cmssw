import FWCore.ParameterSet.Config as cms

from RecoPixelVertexing.PixelTrackFitting.PixelFitterByHelixProjections_cfi import *
from RecoTracker.TkTrackingRegions.GlobalTrackingRegion_cfi import *
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
        RegionPSetBlock,
        ComponentName = cms.string('GlobalRegionProducer')
    ),
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardHitTripletGenerator'),
        SeedingLayers = cms.string('PixelLayerTriplets'),
        GeneratorPSet = cms.PSet(
            PixelTripletHLTGenerator
        )
    ),
    SeedMergerPSet = cms.PSet(
        # layer list for the merger, as defined in (or modified from):
        # RecoPixelVertexing/PixelTriplets/python/quadrupletseedmerging_cff.py
        layerListName = cms.string( "PixelSeedMergerQuadruplets" ),
        # merge triplets -> quadruplets if applicable?
        mergeTriplets = cms.bool( True ),
        # add remaining (non-merged) triplets to merged output quadruplets?
        # (results in a "mixed" output)
        addRemainingTriplets = cms.bool( False ),
        # the builder
        ttrhBuilderLabel = cms.string( "TTRHBuilderPixelOnly" )
    ),
    CleanerPSet = cms.PSet(
        ComponentName = cms.string('PixelTrackCleanerBySharedHits')
    )
)

