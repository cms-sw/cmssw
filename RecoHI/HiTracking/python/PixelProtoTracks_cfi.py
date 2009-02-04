import FWCore.ParameterSet.Config as cms

pixel3ProtoTracks = cms.EDFilter("PixelTrackProducerWithZPos",
    FilterPSet = cms.PSet(
        ComponentName = cms.string('ClusterShapeTrackFilter')
    ),
    passLabel = cms.string(''),
    FitterPSet = cms.PSet(
        ComponentName = cms.string('TrackFitter'),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelTriplets')
    ),
    RegionFactoryPSet = cms.PSet(
        ComponentName = cms.string('HITrackingRegionProducer'),
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            originHalfLength = cms.double(15.9),
            originZPos = cms.double(0.0),
            originYPos = cms.double(0.0),
            directionXCoord = cms.double(1.0),
            directionZCoord = cms.double(0.0),
            VertexCollection = cms.string('pixelVertices'),
            ptMin = cms.double(0.5),
            originXPos = cms.double(0.0),
            useFoundVertices = cms.bool(False),
            directionYCoord = cms.double(1.0),
            originRadius = cms.double(0.1)
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



