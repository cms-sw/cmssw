import FWCore.ParameterSet.Config as cms

pixelLowPtTracksWithZPos = cms.EDFilter("PixelTrackProducerWithZPos",
    # Filter
    FilterPSet = cms.PSet(
        ComponentName = cms.string('ClusterShapeTrackFilter')
    ),
    # Pass
    passLabel = cms.string(''),
    # Fitter
    FitterPSet = cms.PSet(
        ComponentName = cms.string('LowPtPixelFitterByHelixProjections'),
        TTRHBuilder = cms.string('PixelTTRHBuilderWithoutAngle')
    ),
    # Region
    RegionFactoryPSet = cms.PSet(
        ComponentName = cms.string('GlobalRegionProducerWithVertices'),
        RegionPSet = cms.PSet(
            precise = cms.bool(False),
            useChi2Cut = cms.bool(False),
            originHalfLength = cms.double(15.9), ## cm

            originRadius = cms.double(0.2), ## cm

            ptMin = cms.double(0.075), ## GeV/c

            useFoundVertices = cms.bool(False),
            originZPos = cms.double(0.0) ## cm

        )
    ),
    # Cleaner
    CleanerPSet = cms.PSet(
        ComponentName = cms.string('LowPtPixelTrackCleanerBySharedHits')
    ),
    # Generator
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardHitTripletGenerator'),
        SeedingLayers = cms.string('PixelLayerTriplets'),
        GeneratorPSet = cms.PSet(
            ComponentName = cms.string('PixelTripletLowPtGenerator'),
            useClusterShape = cms.bool(True),
            useCleanerMerger = cms.bool(True)
        )
    )
)


