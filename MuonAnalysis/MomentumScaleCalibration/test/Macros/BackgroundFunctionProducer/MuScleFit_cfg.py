import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.CommonDetUnit.globalTrackingGeometry_cfi")
process.load("RecoMuon.DetLayers.muonDetLayerGeometry_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("RecoMuon.TrackingTools.MuonServiceProxy_cff")
process.load("Alignment.CommonAlignmentProducer.GlobalPosition_Fake_cff")
process.load("MagneticField.Engine.uniformMagneticField_cfi")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(0)
)

process.looper = cms.Looper(
    "MuScleFit",
    process.MuonServiceProxy,

    # Choose the kind of muons you want to run on
    # -------------------------------------------

    # // global muons //
    # MuonLabel = cms.InputTag("muons"),
    # MuonType = cms.int32(1),

    # // inner track //
    # MuonLabel = cms.InputTag("muons"),
    # MuonType = cms.int32(10),

    # // standalone muons //
    # MuonLabel = cms.InputTag("standAloneMuons:UpdatedAtVtx"),
    # muonType = cms.int32(2),

    # // tracker tracks //
    # MuonLabel = cms.InputTag("generalTracks"), # ctfWithMaterialTracks
    # muonType = cms.int32(3),

    # // Onia cuts //
    MuonLabel = cms.InputTag("patMuons"),
    MuonType = cms.int32(-1),

    SeparateRanges = cms.untracked.bool(SEPARATERANGES),
    MaxMuonPt = cms.untracked.double(100000000.),
    MinMuonPt = cms.untracked.double(0.),
    MinMuonEtaFirstRange = cms.untracked.double(MIN_MUONETAFIRSTRANGE),
    MaxMuonEtaFirstRange = cms.untracked.double(MAX_MUONETAFIRSTRANGE),
    MinMuonEtaSecondRange = cms.untracked.double(MIN_MUONETASECONDRANGE),
    MaxMuonEtaSecondRange = cms.untracked.double(MAX_MUONETASECONDRANGE),

    # Output settings
    # ---------------
    OutputFileName = cms.untracked.string("OUTPUTNAME"),
    debug = cms.untracked.int32(0),

    # Likelihood settings
    # -------------------
    maxLoopNumber = cms.untracked.int32(1),
    # Select which fits to do in which loop (0 = do not, 1 = do)
    doResolFit =        cms.vint32(0),
    doScaleFit =        cms.vint32(0),
    doBackgroundFit =   cms.vint32(0),
    doCrossSectionFit = cms.vint32(0),

    # Fit parameters and fix flags (1 = use par)
    # ==========================================

    # BiasType=0 means no bias to muon momenta
    # ----------------------------------------
    BiasType = cms.int32(0),
    parBias = cms.vdouble(),

    # BiasType = 1 means linear bias on the muons Pt
    # the two parameters are the constant and the Pt
    # coefficient in this order.
    # ----------------------------------------------
    # BiasType = cms.int32(1),
    # parBias = cms.vdouble(1.001, 0.),

    # Sinusoidal in phi
    # -----------------
    # BiasType = 3 means sinusoidal bias on the muons Pt
    # the two parameters are defined by:
    # pt = (parScale[0] + parScale[1]*sin(phi))*pt; 
    # BiasType = cms.int32(3),
    # parBias = cms.vdouble(1.015, 0.025),

    # SmearType=0 means no smearing applied to muon momenta
    # -----------------------------------------------------
    SmearType = cms.int32(0),
    parSmear = cms.vdouble(),

    # ------------------------- #
    # Resolution fit parameters #
    # ------------------------- #
    ResolFitType  = cms.int32(30), #inner tracks in 31X
    parResol = cms.vdouble(
    0.70,            ##  0 ##
    0.005812,        ##  1 ##
    0.,	             ##  2 ##
    0.0058,          ##  3 ##
    0.012054,        ##  4 ##
    0.,	             ##  5 ##
    0.007135,        ##  6 ##
    0.7619,          ##  7 ##
    0.000990502,     ##  8 ##
    -0.00023,        ##  9 ##
    -0.0005,         ## 10 ##
    0.00107,         ## 11 ##
    0.00000005,      ## 12 ##
    0.0001838,       ## 13 ##
    1.844,           ## 14 ##
    2.1887,          ## 15 ##
    0.027957,        ## 16 ##
    0.,	             ## 17 ##
    0.27,            ## 18 ##
    2.2871,          ## 19 ##
    0.,	             ## 20 ##
    0.,	             ## 21 ##
    0.,	             ## 22 ##
    0.,	             ## 23 ##
    0.,	             ## 24 ##
    0.,	             ## 25 ##
    0.,              ## 26 ##
    ),
    parResolFix = cms.vint32(
    0,    ##  0 ##
    0,    ##  1 ##
    1,    ##  2 ##
    0,    ##  3 ##
    0,    ##  4 ##
    1,    ##  5 ##
    0,    ##  6 ##
    0,    ##  7 ##
    1,    ##  8 ##
    1,    ##  9 ##
    1,    ## 10 ##
    1,    ## 11 ##
    0,    ## 12 ##
    0,    ## 13 ##
    0,    ## 14 ##
    0,    ## 15 ##
    0,    ## 16 ##
    1,    ## 17 ##
    0,    ## 18 ##
    0,    ## 19 ##
    1,    ## 20 ##
    1,    ## 21 ##
    1,    ## 22 ##
    1,    ## 23 ##
    1,    ## 24 ##
    1,    ## 25 ##
    1     ## 26 ##
    ),
    parResolOrder = cms.vint32(
    0,    ##  0 ##
    0,    ##  1 ##
    0,    ##  2 ##
    0,    ##  3 ##
    0,    ##  4 ##
    0,    ##  5 ##
    0,    ##  6 ##
    0,    ##  7 ##
    0,    ##  8 ##
    0,    ##  9 ##
    0,    ## 10 ##
    0,    ## 11 ##
    0,    ## 12 ##
    0,    ## 13 ##
    0,    ## 14 ##
    0,    ## 15 ##
    0,    ## 16 ##
    0,    ## 17 ##
    0,    ## 18 ##
    0,    ## 19 ##
    0,    ## 20 ##
    0,    ## 21 ##
    0,    ## 22 ##
    0,    ## 23 ##
    0,    ## 24 ##
    0,    ## 25 ##
    0     ## 26 ##
    ),

    # -------------------- #
    # Scale fit parameters #
    # -------------------- #
    ScaleFitType = cms.int32(23),
    parScale = cms.vdouble(
    1.0009,         ##  0 ##
    -0.00020,       ##  1 ##
    0.79,           ##  2 ##
    -0.00045,       ##  3 ##
    -0.44,          ##  4 ##
    -0.00017,       ##  5 ##
    1.7,            ##  6 ##
    -0.0006,        ##  7 ##
    0.34,           ##  8 ##
    0.47,           ##  9 ##
    0.,             ## 10 ##
    ),
    parScaleFix =   cms.vint32(
    0,           ##  0 ##
    0,           ##  1 ##
    0,           ##  2 ##
    0,           ##  3 ##
    0,           ##  4 ##
    0,           ##  5 ##
    0,           ##  6 ##
    0,           ##  7 ##
    0,           ##  8 ##
    0,           ##  9 ##
    1,           ## 10 ##
    ),
    parScaleOrder = cms.vint32(
    0,           ##  0 ##
    0,           ##  1 ##
    0,           ##  2 ##
    0,           ##  3 ##
    0,           ##  4 ##
    0,           ##  5 ##
    0,           ##  6 ##
    0,           ##  7 ##
    0,           ##  8 ##
    0,           ##  9 ##
    0,           ## 10 ##
    ),

    # ---------------------------- #
    # Cross section fit parameters #
    # ---------------------------- #
    parCrossSectionOrder = cms.vint32(0, 0, 0, 0, 0, 0),
    parCrossSectionFix =   cms.vint32(0, 0, 0, 0, 0, 0),
    parCrossSection = cms.vdouble(1.233, 2.07, 6.33, 13.9, 2.169, 127.2),

    # ------------------------- #
    # Background fit parameters #
    # ------------------------- #
    # Window factors for: Z, Upsilons and (J/Psi,Psi2S) regions
    LeftWindowBorder = cms.vdouble(70., 8., 2.6),
    RightWindowBorder = cms.vdouble(110., 12., 4.),

    # The two parameters of BgrFitType=2 are respectively:
    # bgr fraction, (negative of) bgr exp. slope, bgr constant
    # --------------------------------------------------------
    # The function types for resonances in a region must be the same

    # ------------------------------------------- #
    # ATTENTION: careful with the parameters, the background probability is always computed, putting
    # parameters for which the function will return nan will do so that the background probability will
    # return 0, however it is not a good thing to do.
    # ------------------------------------------- #

    BgrFitType = cms.vint32(2, 2, 2), # resonances
    parBgr = cms.vdouble(0., 0.,   0., 0.,   0.621704, 0.6819,
                         0., 0.,   0., 0.,   0., 0.,     0.,0.,   0.,0.,   0.,0.),
    parBgrFix = cms.vint32(1, 1,   0, 0,   0, 0,
                           # The rest of the parameters is used for the resonance regions. They are automatically fixed in the code
                           # because they are never used to fit the background, but only after the rescaling.
                           1, 1,   1, 1,   1, 1,   1, 1,   1, 1,   1, 1),
    parBgrOrder = cms.vint32(0, 0,   0, 0,   0, 0,
                             0, 0,   0, 0,   0, 0,   0, 0,   0, 0,   0, 0),

    # ---------------- #
    # Select resonance #
    # ---------------- #

    # The resonances are to be specified in this order:
    # Z0, Y(3S), Y(2S), Y(1S), Psi(2S), J/Psi
    # -------------------------------------------------
    resfind = cms.vint32(0, 0, 0, 0, 1, 1),
    FitStrategy = cms.int32(2),

    speedup = cms.bool(True),
    # Set this to false if you do not want to use simTracks.
    # (Note that this is skipped anyway if speedup == True).
    compareToSimTracks = cms.bool(False),
    Sherpa = cms.untracked.bool(False),
    DebugMassResol = cms.untracked.bool(False),
    # readPdfFromDB = cms.bool(False)
    PATmuons = cms.untracked.bool(True),
    genParticles = cms.bool(False),
    GenParticlesName = cms.untracked.string('generator'),
    HepMCProduct = cms.bool(True),

    ProbabilitiesFile = cms.untracked.string("/home/demattia/FSR/CMSSW_3_6_1_patch4/src/MuonAnalysis/MomentumScaleCalibration/test/Probs_merge.root"),

    # Only used when reading events from a root tree
    MaxEventsFromRootTree = cms.int32(-1),

    #InputRootTreeFileName = cms.string("/home/demattia/MuScleFit/TreeConversion/NewTree/newTree_oniaSel_upTo148068_19invpb.root"),
    InputRootTreeFileName = cms.string("/home/demattia/MuScleFit/TreeConversion/NewTree/newTree_oniaSel_upTo148068_19invpb.root"),
    OutputRootTreeFileName = cms.string(""),

    # Fit accuracy and debug parameters
    StartWithSimplex = cms.bool(True),
    ComputeMinosErrors = cms.bool(False),
    MinimumShapePlots = cms.bool(False),

    # The following parameters can be used to filter events
    TriggerResultsLabel = cms.untracked.string("TriggerResults"),
    TriggerResultsProcess = cms.untracked.string(""),
    # TriggerPath: "" = No trigger requirements, "All" = No specific path
    #TriggerPath = cms.untracked.string("HLT_L1MuOpen"),
    TriggerPath = cms.untracked.string(""),
    # Negate the result of the trigger
    NegateTrigger = cms.untracked.bool(False),
)

# Timing information
# process.load("FWCore.MessageLogger.MessageLogger_cfi")
# TimingLogFile = cms.untracked.string('timing.log')
# process.Timing = cms.Service("Timing")


