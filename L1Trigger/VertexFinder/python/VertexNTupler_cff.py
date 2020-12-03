import FWCore.ParameterSet.Config as cms

L1TVertexNTupler = cms.EDAnalyzer('VertexNTupler',
  tpInputTag = cms.InputTag("mix", "MergedTrackTruth"),
  stubInputTag = cms.InputTag("TTStubsFromPhase2TrackerDigis", "StubAccepted"),
  stubTruthInputTag = cms.InputTag("TTStubAssociatorFromPixelDigis", "StubAccepted"),
  clusterTruthInputTag = cms.InputTag("TTClusterAssociatorFromPixelDigis", "ClusterAccepted"),

  l1TracksInputTags    = cms.VInputTag( cms.InputTag("TTTracksFromTrackletEmulation", "Level1TTTracks") ),
  l1TracksTruthMapInputTags = cms.VInputTag( cms.InputTag("TTTrackAssociatorFromPixelDigis", "Level1TTTracks") ),
  l1TracksBranchNames  = cms.vstring('hybrid'),
  l1VertexInputTags   = cms.VInputTag( cms.InputTag("VertexProducer", "l1vertices") ),
  l1VertexTrackInputs = cms.vstring('hybrid'),
  l1VertexBranchNames = cms.vstring('dbscan'),
  extraL1VertexInputTags = cms.VInputTag(),
  extraL1VertexDescriptions = cms.vstring(),

  #=== Cuts on MC truth particles (i.e., tracking particles) used for tracking efficiency measurements.
  GenCuts = cms.PSet(
     GenMinPt         = cms.double(2.0),
     GenMaxAbsEta     = cms.double(2.4),
     GenMaxVertR      = cms.double(1.0), # Maximum distance of particle production vertex from centre of CMS.
     GenMaxVertZ      = cms.double(30.0),
     GenPdgIds        = cms.vuint32(), # Only particles with these PDG codes used for efficiency measurement.


     # Additional cut on MC truth tracks used for algorithmic tracking efficiency measurements.
     # You should usually set this equal to value of L1TrackDef.MinStubLayers below, unless L1TrackDef.MinPtToReduceLayers
     # is < 10000, in which case, set it equal to (L1TrackDef.MinStubLayers - 1).
     GenMinStubLayers = cms.uint32(4)
  ),


  #=== Rules for deciding when the track finding has found an L1 track candidate
  L1TrackDef = cms.PSet(
     UseLayerID           = cms.bool(True),
     # Reduce this layer ID, so that it takes no more than 8 different values in any eta region (simplifies firmware).
     ReducedLayerID       = cms.bool(True)
  ),

  #=== Rules for deciding when a reconstructed L1 track matches a MC truth particle (i.e. tracking particle).
  TrackMatchDef = cms.PSet(
     #--- Three different ways to define if a tracking particle matches a reco track candidate. (Usually, set two of them to ultra loose).
     # Min. fraction of matched stubs relative to number of stubs on reco track.
     MinFracMatchStubsOnReco  = cms.double(-99.),
     # Min. fraction of matched stubs relative to number of stubs on tracking particle.
     MinFracMatchStubsOnTP    = cms.double(-99.),
     # Min. number of matched layers.
     MinNumMatchLayers        = cms.uint32(4),
     # Min. number of matched PS layers.
     MinNumMatchPSLayers      = cms.uint32(0),
     # Associate stub to TP only if the TP contributed to both its clusters? (If False, then associate even if only one cluster was made by TP).
     StubMatchStrict          = cms.bool(False)
  ),


  # === Vertex Reconstruction configuration
  VertexReconstruction=cms.PSet(
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
        # FastHisto algorithm assumed vertex width [cm]
        TP_VertexWidth = cms.double(.15),
        # Kmeans number of iterations
        KmeansIterations = cms.uint32(10),
        # Kmeans number of clusters
        KmeansNumClusters  = cms.uint32(18),
        # DBSCAN pt threshold
        DBSCANPtThreshold = cms.double(4.),
        # DBSCAN min density tracks
        DBSCANMinDensityTracks = cms.uint32(2),
        VxMinTrackPt   = cms.double(2.5)
    ),

  # Debug printout
  debug  = cms.uint32(0),
  printResults = cms.bool(False)
)
