import FWCore.ParameterSet.Config as cms

allPixelTracks = cms.EDFilter("PixelTrackProducerWithZPos",
    FilterPSet = cms.PSet(
        ComponentName = cms.string('ClusterShapeTrackFilter')
    ),
    passLabel = cms.string(''),
    FitterPSet = cms.PSet(
        ComponentName = cms.string('TrackFitter'),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelTriplets')
    ),
    RegionFactoryPSet = cms.PSet(
        ComponentName = cms.string('GlobalTrackingRegionWithVerticesProducer'),
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            beamSpot = cms.InputTag("offlineBeamSpot"),
            useFixedError = cms.bool(True),
            originRadius = cms.double(0.2),
            sigmaZVertex = cms.double(3.0),
            fixedError = cms.double(0.2),
            VertexCollection = cms.string('pixelVertices'),
            ptMin = cms.double(0.075),
            useFoundVertices = cms.bool(False),
            nSigmaZ = cms.double(3.0)
        )
    ),
    CleanerPSet = cms.PSet(
        ComponentName = cms.string('TrackCleaner')
    ),
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardHitTripletGenerator'),
        SeedingLayers = cms.string('PixelLayerTriplets'),
        GeneratorPSet = cms.PSet(
            nSigMultipleScattering = cms.double(5.0),
            checkMultipleScattering = cms.bool(True),
            ComponentName = cms.string('TripletGenerator'),
            checkClusterShape = cms.bool(True),
            maxAngleRatio = cms.double(10.0),
            rzTolerance = cms.double(0.2)
        )
    )
)


