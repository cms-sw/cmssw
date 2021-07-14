import FWCore.ParameterSet.Config as cms
import RecoVertex.PrimaryVertexProducer.primaryVertexProducer_cfi as _mod

hiPixelAdaptiveVertex = _mod.primaryVertexProducer.clone(
    verbose = False,
    TkFilterParameters = dict(
        algorithm = 'filterWithThreshold',
        maxNormalizedChi2 = 5.0,
        minSiliconLayersWithHits = 0, ## >=0 (was 5 for generalTracks)
        minPixelLayersWithHits = 2,   ## >=2 (was 2 for generalTracks)
        maxD0Significance = 3.0,     ## keep most primary tracks (was 5.0)
        minPt = 0.0,                 ## better for softish events
        maxEta = 100.,
        numTracksThreshold = 2
    ),
    # label of tracks to be used
    TrackLabel = "hiSelectedProtoTracks",
    # clustering
    TkClusParameters = dict(
        algorithm = "gap",
        TkGapClusParameters = cms.PSet(
            zSeparation = cms.double(1.0)       ## 1 cm max separation between clusters
        )
    ),
    vertexCollections = {
        0: dict(
        label = '',
        chi2cutoff = 3.0,
        maxDistanceToBeam = 0.1
        )
    }
)
