import FWCore.ParameterSet.Config as cms

l1tVertexNTupler = cms.EDAnalyzer('VertexNTupler',
  inputDataInputTag = cms.InputTag("l1tInputDataProducer","InputData"),
  genParticleInputTag = cms.InputTag("genParticles",""),
  l1TracksInputTags    = cms.VInputTag( cms.InputTag("l1tTTTracksFromTrackletEmulation", "Level1TTTracks") ),
  l1TracksTruthMapInputTags = cms.VInputTag( cms.InputTag("TTTrackAssociatorFromPixelDigis", "Level1TTTracks") ),
  l1TracksTPInputTags = cms.InputTag("l1tTPStubValueMapProducer:allMatchedTPs"),
  l1TracksTPValueMapInputTags = cms.InputTag("l1tTPStubValueMapProducer:TPs"),
  l1TracksBranchNames  = cms.vstring('hybrid'),
  l1VertexInputTags   = cms.VInputTag( cms.InputTag("l1tVertexProducer", "L1Vertices") ),
  l1VertexTrackInputs = cms.vstring('hybrid'),
  l1VertexBranchNames = cms.vstring('fastHisto'),
  emulationVertexInputTags = cms.VInputTag(),
  emulationVertexBranchNames = cms.vstring(),
  extraL1VertexInputTags = cms.VInputTag(),
  extraL1VertexDescriptions = cms.vstring(),

  genJetsInputTag = cms.InputTag("ak4GenJetsNoNu"),

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
  VertexReconstruction = cms.PSet(
    # Vertex Reconstruction Algorithm
    Algorithm = cms.string("fastHisto"),
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
    # Do track quality cuts in emulation algorithms
    EM_DoQualityCuts = cms.bool(False),
    # Track-stubs Pt compatibility cut
    FH_DoPtComp = cms.bool(True),
    # chi2dof < 5 for tracks with Pt > 10
    FH_DoTightChi2 = cms.bool(False),
    # fastHisto algorithm histogram parameters (min,max,width) [cm]
    # TDR settings: [-14.95, 15.0, 0.1]
    # L1TkPrimaryVertexProducer: [-30.0, 30.0, 0.09983361065]
    # Firmware: [-14.4, 14.4, 0.4]
    FH_HistogramParameters = cms.vdouble(-30.0, 30.0, 0.09983361065),
    # The number of vertixes to return (i.e. N windows with the highest combined pT)
    FH_NVtx = cms.uint32(10),
    # fastHisto algorithm assumed vertex half-width [cm]
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
    VxMaxTrackPt = cms.double(127.0),
    # When the track pt > VxMaxTrackPt, how should the tracks be considered
    #   -1 = tracks are valid
    #   0 = tracks are mismeasured and ignored/truncated
    #   1 = tracks are mismeasured and saturate at VxMaxTrackPt
    # Option '0' was used for the TDR, but '1' is used for the firmware
    VxMaxTrackPtBehavior = cms.int32(1),
    # Maximum chi2 of tracks used to create vertex
    VxMaxTrackChi2 = cms.double(100.),
    # Minimum number of stubs associated to a track
    VxMinNStub = cms.uint32(4),
    # Minimum number of stubs in PS modules associated to a track
    VxMinNStubPS = cms.uint32(3),
  ),

  # Debug printout
  debug  = cms.uint32(0),
  printResults = cms.bool(False)
)
