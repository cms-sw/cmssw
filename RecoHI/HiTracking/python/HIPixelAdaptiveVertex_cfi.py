import FWCore.ParameterSet.Config as cms
from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi import offlinePrimaryVertices

hiPixelAdaptiveVertex = offlinePrimaryVertices.clone(
    verbose = False,
    TkFilterParameters = dict(
        algorithm = 'filterWithThreshold',
        maxNormalizedChi2 = 5.0,
        minSiliconLayersWithHits = 0, ## >=0 (was 5 for generalTracks)
        minPixelLayersWithHits = 2,   ## >=2 (was 2 for generalTracks)
        maxD0Significance = 3.0,     ## keep most primary tracks (was 5.0)
        minPt = 0.0,                 ## better for softish events
        maxEta = 100.,
    ),
    # label of tracks to be used
    TrackLabel = "hiSelectedProtoTracks",
    # clustering
    TkClusParameters = cms.PSet(
        algorithm = cms.string("gap"),
        TkGapClusParameters = cms.PSet(
            zSeparation = cms.double(1.0)       ## 1 cm max separation between clusters
        )
    ),
    vertexCollections = cms.VPSet(
      cms.PSet(
        label = cms.string(''),
        chi2cutoff = cms.double(3.0),
        algorithm = cms.string('AdaptiveVertexFitter'),
        useBeamConstraint = cms.bool(False),
        maxDistanceToBeam = cms.double(0.1),
        minNdof  = cms.double(0.0)
        )
      )
)

hiPixelAdaptiveVertex.TkFilterParameters.numTracksThreshold = cms.int32(2)
