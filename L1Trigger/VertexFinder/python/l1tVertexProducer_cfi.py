import FWCore.ParameterSet.Config as cms

l1tVertexProducer = cms.EDProducer('VertexProducer',                                   
  l1TracksInputTag = cms.InputTag("l1tTrackSelectionProducer", "Level1TTTracksSelected"),
                                   
  l1VertexCollectionName = cms.string("L1Vertices"), #Emulation postfix is appended when fastHistoEmulation is chosen as the algorithm

  # === Vertex Reconstruction configuration
  VertexReconstruction = cms.PSet(
        # Vertex Reconstruction Algorithm
        Algorithm = cms.string("PFA"),
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
        WeightedMean = cms.uint32(2),
        # Chi2 cut for the Adaptive Vertex Reconstruction Algorithm
        AVR_chi2cut = cms.double(5.),
        # Do track quality cuts in emulation algorithms
        EM_DoQualityCuts = cms.bool(False),
        # Track-stubs Pt compatibility cut
        FH_DoPtComp = cms.bool(False),
        # chi2dof < 5 for tracks with Pt > 10
        FH_DoTightChi2 = cms.bool(False),
        # PFA algorithm scan parameters (min,max,width) [cm]
        PFA_ScanParameters = cms.vdouble(-20.46912512, 20.46912512, 0.15991504),
        # Include eta-dependence of the estimated track resolution used in PFA
        PFA_EtaDependentResolution = cms.bool(True),
        # Scale factor for the PFA track resolution parameter (where the nominal values with and without eta-dependence are hard-coded using the fit results from Giovanna's thesis)
        PFA_ResolutionSF = cms.double(2.),
        # PFA Gaussian width cutoff [cm]
        PFA_VertexWidth = cms.double(1.31), # Giovanna's recommendation of 3*sigma(lowest-resolution tracks).
        # Enable 2-step process where the weighted pT sum is only calculated at positions where the weighted multiplicity is maximum ("local maxima"). In the second step, the local maximum with the largest weighted pT sum is chosen as the vertex.
        PFA_UseMultiplicityMaxima = cms.bool(False),
        # Weight function to use in PFA. 0: Gaussian, 1: Gaussian without width normalisation, 2: Complementary error function, 3: Step function
        PFA_WeightFunction = cms.uint32(1),
        # Instead of taking the z0 value from the discrete PFA scan (0), calculate it from the Gaussian and pT-weighted average of track z0 (1) or the optimal (1/variance) weighted mean of associated tracks, weighted also by pT and association probability (2). Step function and pT-weighted average (3) is only designed for use with PFA_WeightFunction=3 (to replicate fastHisto).
        PFA_WeightedZ0 = cms.uint32(1),
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
        VxMaxTrackChi2 = cms.double(99999999.),
        # Minimum number of stubs associated to a track
        VxMinNStub = cms.uint32(0),
        # Minimum number of stubs in PS modules associated to a track
        VxMinNStubPS = cms.uint32(0),
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
