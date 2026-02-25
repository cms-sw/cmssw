import FWCore.ParameterSet.Config as cms

l1tVertexProducer = cms.EDProducer('VertexProducer',                                   
  l1TracksInputTag = cms.InputTag("l1tTrackSelectionProducer", "Level1TTTracksSelected"),
                                   
  l1VertexCollectionName = cms.string("L1Vertices"), #Emulation postfix is appended when fastHistoEmulation is chosen as the algorithm

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
        # PFA algorithm scan parameters (min,max,interval) [cm]
        PFA_ScanParameters = cms.vdouble(-20.46912512, 20.46912512, 0.15991504), # set to the same values as for FH for ease of comparison
        # Include eta-dependence of the estimated track resolution used in PFA
        PFA_EtaDependentResolution = cms.bool(True),
        # Multiplicative scale factor for the above PFA track resolution parameter
        PFA_ResolutionSF = cms.double(1.3),
        # PFA Gaussian width cutoff for input tracks [cm] (not used in PFASimple)
        PFA_Cutoff = cms.double(1.31), # Using recommendation of 3*sigma(lowest-resolution tracks) from CERN-THESIS-2024-143
        # Enable 2-step process where the weighted pT sum is only calculated at positions where the weighted multiplicity is maximum ("local maxima"). In the second step, the local maximum with the largest weighted pT sum is chosen as the vertex. Only relevant for PFA (not used in PFASimple).
        PFA_UseMultiplicityMaxima = cms.bool(False),
        # Weight function to use in PFA (not used in PFASimple). 0: Gaussian, 1: Gaussian without width normalisation, 2: Complementary error function, 3: Step function. With PFA_WeightFunction=3 and PFA_UseMultiplicityMaxima=False, PFA and PFASimple are the same.
        PFA_WeightFunction = cms.uint32(3),
        # Instead of taking the z0 value from the discrete PFA scan (0), calculate it from the Gaussian and pT^N-weighted average of track z0 (1) or the optimal (1/variance) weighted mean of associated tracks, weighted also by pT^N and association probability (2). Step function and pT^N-weighted average (3) is intended for use with PFA_WeightFunction=3 or PFASimple (to replicate fastHisto).
        # Additional options (4-11) have different uses of the track resolution (only relevant when eta-dependent) and different powers of trackPt in the weighted sum: see VertexFinder.cc for details.
        PFA_WeightedZ0 = cms.uint32(10), # 10 seems best overall for PFA when using WeightedMean=2, but 7-11 all have similar performance (with the best one depending on the process)
        # Use VxMinTrackPt cut specified below (otherwise no additional track selection is applied)
        PFA_DoQualityCuts = cms.bool(False),
        # fastHisto algorithm histogram parameters (min,max,width) [cm]
        # TDR settings: [-14.95, 15.0, 0.1]
        # L1TkPrimaryVertexProducer: [-30.0, 30.0, 0.09983361065]
        # HLS Firmware: [-14.4, 14.4, 0.4]
        # Track word limits (128 binns): [-20.46912512, 20.46912512, 0.31983008]
        # Track word limits (256 binns): [-20.46912512, 20.46912512, 0.15991504]
        FH_HistogramParameters = cms.vdouble(-20.46912512, 20.46912512, 0.15991504),
        # The number of vertixes to return (i.e. N windows with the highest combined pT)
        FH_NVtx = cms.uint32(1),
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
        # Track weight NN graph 
        TrackWeightGraph = cms.FileInPath("L1Trigger/VertexFinder/data/NNVtx_WeightModelGraph.pb"),
        # Pattern recognition NN graph
        PatternRecGraph = cms.FileInPath("L1Trigger/VertexFinder/data/NNVtx_PatternModelGraph.pb"),
    ),
  # Debug printout
  debug  = cms.uint32(0)
)

l1tVertexFinder = l1tVertexProducer.clone()

l1tVertexFinderEmulator = l1tVertexProducer.clone()
l1tVertexFinderEmulator.VertexReconstruction.Algorithm = cms.string("fastHistoEmulation")
l1tVertexFinderEmulator.l1TracksInputTag = cms.InputTag("l1tTrackSelectionProducer", "Level1TTTracksSelectedEmulation")
