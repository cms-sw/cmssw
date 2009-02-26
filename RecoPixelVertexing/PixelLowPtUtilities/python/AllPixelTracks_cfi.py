import FWCore.ParameterSet.Config as cms

##########################
# The base for all steps
allPixelTracks = cms.EDFilter("PixelTrackProducerWithZPos",

    passLabel  = cms.string(''),

    # Region
    RegionFactoryPSet = cms.PSet(
        ComponentName = cms.string('GlobalTrackingRegionWithVerticesProducer'),
        RegionPSet = cms.PSet(
            precise       = cms.bool(True),
            beamSpot      = cms.InputTag("offlineBeamSpot"),
            originRadius  = cms.double(0.2),
            sigmaZVertex  = cms.double(3.0),
            useFixedError = cms.bool(True),
            fixedError    = cms.double(0.2),

            useFoundVertices = cms.bool(False),
            VertexCollection = cms.InputTag("pixelVertices"),
            ptMin            = cms.double(0.075),
            nSigmaZ          = cms.double(3.0)
        )
    ),
     
    # Ordered hits
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardHitTripletGenerator'),
        SeedingLayers = cms.string('PixelLayerTriplets'),
        GeneratorPSet = cms.PSet(
            ComponentName = cms.string('TripletGenerator'),
            checkClusterShape       = cms.bool(False),
            checkMultipleScattering = cms.bool(True),
            nSigMultipleScattering  = cms.double(5.0),
            maxAngleRatio = cms.double(10.0),
            rzTolerance   = cms.double(0.2),
            TTRHBuilder   = cms.string('TTRHBuilderWithoutAngle4PixelTriplets')
        )
    ),

    # Filter
    FilterPSet = cms.PSet(
        ComponentName = cms.string('ClusterShapeTrackFilter'),
        useFilter     = cms.bool(True)
    ),

    # Cleaner
    CleanerPSet = cms.PSet(
        ComponentName = cms.string('TrackCleaner')
    ),

    # Fitter
    FitterPSet = cms.PSet(
        ComponentName = cms.string('TrackFitter'),
        TTRHBuilder   = cms.string('TTRHBuilderWithoutAngle4PixelTriplets')
    )
)


