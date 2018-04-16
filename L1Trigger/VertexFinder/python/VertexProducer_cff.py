import FWCore.ParameterSet.Config as cms

VertexProducer = cms.EDProducer('VertexProducer',

  l1TracksInputTag = cms.InputTag("TMTrackProducer", "TML1TracksSimpleLR"), # SFLR

  # === Vertex Reconstruction configuration
  VertexReconstruction=cms.PSet(
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
        WeightedMean = cms.bool(True),
        # Chi2 cut for the Adaptive Vertex Reconstruction Algorithm
        AVR_chi2cut = cms.double(5.),
        # TDR algorithm assumed vertex half-width [cm]
        TP_VertexWidth = cms.double(.15),
        # Kmeans number of iterations
        KmeansIterations = cms.uint32(10),
        # Kmeans number of clusters
        KmeansNumClusters  = cms.uint32(18),
        # DBSCAN pt threshold
        DBSCANPtThreshold = cms.double(2.),
        # DBSCAN min density tracks
        DBSCANMinDensityTracks = cms.uint32(1),

        VxMinTrackPt   = cms.double(2.)
    ),
  # Debug printout
  Debug  = cms.uint32(0), #(0=none, 1=print tracks/sec, 2=show filled cells in HT array in each sector of each event, 3=print all HT cells each TP is found in, to look for duplicates, 4=print missed tracking particles by r-z filters, 5 = show debug info about duplicate track removal, 6 = show debug info about fitters)
)
