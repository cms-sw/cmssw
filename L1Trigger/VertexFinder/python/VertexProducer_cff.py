import FWCore.ParameterSet.Config as cms

VertexProducer = cms.EDProducer('VertexProducer',

  l1TracksInputTag = cms.InputTag("TTTracksFromTrackletEmulation", "Level1TTTracks"),

  l1VertexCollectionName = cms.string("l1vertices"),

  # === Vertex Reconstruction configuration
  VertexReconstruction = cms.PSet(
        # Vertex Reconstruction Algorithm
        Algorithm = cms.string("DBSCAN"),
        # Vertex distance
        VertexDistance = cms.double(.15),
        # Assumed Vertex Resolution
        VertexResolution = cms.double(.10),
        # Distance Type for agglomerative algorithm (0: MaxDistance, 1: MinDistance, 2: MeanDistance, 3: CentralDistance)
        DistanceType  = cms.uint32(0),
        # Minimum number of tracks to accept vertex
        MinTracks   = cms.uint32(2),
        # Compute the z0 position of the vertex with a mean weighted with track momenta
        WeightedMean = cms.bool(False),
        # Chi2 cut for the Adaptive Vertex Reconstruction Algorithm
        AVR_chi2cut = cms.double(5.),
        # FastHisto algorithm histogram parameters (min,max,width) [cm]
        TP_HistogramParameters = cms.vdouble(-14.95, 15.0, 0.1),
        # FastHisto algorithm assumed vertex half-width [cm]
        TP_VertexWidth = cms.double(.15),
        # Kmeans number of iterations
        KmeansIterations = cms.uint32(10),
        # Kmeans number of clusters
        KmeansNumClusters  = cms.uint32(18),
        # DBSCAN pt threshold
        DBSCANPtThreshold = cms.double(4.),
        # DBSCAN min density tracks
        DBSCANMinDensityTracks = cms.uint32(2),
        # Minimum pt of tracks used to create vertex
        VxMinTrackPt = cms.double(2.0)
    ),
  # Debug printout
  debug  = cms.uint32(0)
)
