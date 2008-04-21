import FWCore.ParameterSet.Config as cms

from RecoPixelVertexing.PixelTrackFitting.PixelFitterByHelixProjections_cfi import *
from RecoTracker.TkTrackingRegions.GlobalTrackingRegion_cfi import *
from RecoPixelVertexing.PixelTriplets.PixelTripletHLTGenerator_cfi import *
pixelTracks = cms.EDProducer("PixelTrackProducer",
    FitterPSet = cms.PSet(
        PixelFitterByHelixProjections
    ),
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
    CleanerPSet = cms.PSet(
        ComponentName = cms.string('PixelTrackCleanerBySharedHits')
    )
)


