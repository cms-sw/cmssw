import FWCore.ParameterSet.Config as cms

VertexProducer = cms.EDProducer('VertexProducer',

  l1TracksInputTag = cms.InputTag("TTTracksFromTrackletEmulation", "Level1TTTracks"),

  l1VertexCollectionName = cms.string("l1vertices"),

  # === Vertex Reconstruction configuration
  VertexReconstruction = cms.PSet(
        # Vertex Reconstruction Algorithm
        Algorithm = cms.string("FastHisto"),
        # Vertex distance [cm]
        VertexDistance = cms.double(.15),
        # Assumed Vertex Resolution [cm]
        VertexResolution = cms.double(.10),
        # Distance Type for agglomerative algorithm (0: MaxDistance, 1: MinDistance, 2: MeanDistance, 3: CentralDistance)
        DistanceType  = cms.uint32(0),
        # Minimum number of tracks to accept vertex
        MinTracks   = cms.uint32(2),
        # Compute the z0 position of the vertex with a mean weighted with track momenta
        #   0 = unweighted
        #   1 = pT weighted
        #   2 = pT^2 weighted
        WeightedMean = cms.uint32(1),
        # Chi2 cut for the Adaptive Vertex Reconstruction Algorithm
        AVR_chi2cut = cms.double(5.),
        # Track-stubs Pt compatibility cut
        FH_DoPtComp = cms.bool(True),
        # chi2dof < 5 for tracks with Pt > 10
        FH_DoTightChi2 = cms.bool(False),
        # FastHisto algorithm histogram parameters (min,max,width) [cm]
        # TDR settings: [-14.95, 15.0, 0.1]
        # L1TkPrimaryVertexProducer: [-30.0, 30.0, 0.09983361065]
        # Firmware: [-14.4, 14.4, 0.4]
        FH_HistogramParameters = cms.vdouble(-30.0, 30.0, 0.09983361065),
        # The number of vertixes to return (i.e. N windows with the highest combined pT)
        FH_NVtx = cms.uint32(10),
        # FastHisto algorithm assumed vertex half-width [cm]
        FH_VertexWidth = cms.double(.15),
        # Window size of the sliding window
        FH_WindowSize = cms.uint32(3),
        # Kmeans number of iterations
        KmeansIterations = cms.uint32(10),
        # Kmeans number of clusters
        KmeansNumClusters  = cms.uint32(18),
        # DBSCAN pt threshold
        DBSCANPtThreshold = cms.double(4.),
        # DBSCAN min density tracks
        DBSCANMinDensityTracks = cms.uint32(2),
        # Minimum pt of tracks used to create vertex [GeV]
        VxMinTrackPt = cms.double(2.0),
        # Maximum pt of tracks used to create vertex [GeV]
        VxMaxTrackPt = cms.double(50.0),
        # When the track pt > VxMaxTrackPt, how should the tracks be considered
        #   -1 = tracks are valid
        #   0 = tracks are mismeasured and ignored/truncated
        #   1 = tracks are mismeasured and saturate at VxMaxTrackPt
        # Option '0' was used for the TDR, but '1' is used for the firmware
        VxMaxTrackPtBehavior = cms.int32(0),
        # Maximum chi2 of tracks used to create vertex
        VxMaxTrackChi2 = cms.double(100.),
        # Minimum number of stubs associated to a track
        VxMinNStub = cms.uint32(4),
        # Minimum number of stubs in PS modules associated to a track
        VxMinNStubPS = cms.uint32(3),
    ),
  # Debug printout
  debug  = cms.uint32(0)
)
