import FWCore.ParameterSet.Config as cms

hiPixelAdaptiveVertex = cms.EDProducer("PrimaryVertexProducer",
    PVSelParameters = cms.PSet(
        maxDistanceToBeam = cms.double(0.1)    ## 1 mm
    ),
    verbose = cms.untracked.bool(False),
    algorithm = cms.string('AdaptiveVertexFitter'),
    minNdof  = cms.double(0.0),
    TkFilterParameters = cms.PSet(
        algorithm = cms.string('filterWithThreshold'),
        maxNormalizedChi2 = cms.double(5.0),
        minSiliconLayersWithHits = cms.int32(0), ## >=0 (was 5 for generalTracks)
        minPixelLayersWithHits = cms.int32(2),   ## >=2 (was 2 for generalTracks)
        maxD0Significance = cms.double(3.0),     ## keep most primary tracks (was 5.0)
        minPt = cms.double(0.0),                 ## better for softish events
        trackQuality = cms.string("any"),
        numTracksThreshold = cms.int32(2)
    ),
    beamSpotLabel = cms.InputTag("offlineBeamSpot"),
    # label of tracks to be used
    TrackLabel = cms.InputTag("hiSelectedProtoTracks"),
    useBeamConstraint = cms.bool(True),
    # clustering
    TkClusParameters = cms.PSet(
        algorithm = cms.string("gap"),
        TkGapClusParameters = cms.PSet(
            zSeparation = cms.double(1.0)        ## 1 cm max separation between clusters
        )
    )
)

