import FWCore.ParameterSet.Config as cms

hiPixelAdaptiveVertex = cms.EDProducer("PrimaryVertexProducer",
    PVSelParameters = cms.PSet(
        maxDistanceToBeam = cms.double(0.02), ## 200 microns

        minVertexFitProb = cms.double(0.01) ## 1% vertex fit probability

    ),
    verbose = cms.untracked.bool(False),
    algorithm = cms.string('AdaptiveVertexFitter'),
    TkFilterParameters = cms.PSet(
        maxNormalizedChi2 = cms.double(5.0),
        minSiliconHits = cms.int32(2), ## hits > 2 (was 7 for generalTracks)

        maxD0Significance = cms.double(3.0), ## keep most primary tracks (was 5.0)

        minPt = cms.double(0.0), ## better for softish events

        minPixelHits = cms.int32(2) ## pixel hits > 2 (was 2 for generalTracks)

    ),
    beamSpotLabel = cms.InputTag("offlineBeamSpot"),
    # label of tracks to be used
    TrackLabel = cms.InputTag("hiSelectedProtoTracks"),
    useBeamConstraint = cms.bool(True),
    VtxFinderParameters = cms.PSet(
        ptCut = cms.double(0.0),
        vtxFitProbCut = cms.double(0.01), ## 1% vertex fit probability
	    trackCompatibilityToSVcut = cms.double(0.01), ## 1%
        trackCompatibilityToPVcut = cms.double(0.05), ## 5%
        maxNbOfVertices = cms.int32(0) ## search all vertices in each cluster

    ),
    TkClusParameters = cms.PSet(
        zSeparation = cms.double(0.1) ## 1 mm max separation betw. clusters

    )
)

