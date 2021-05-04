import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
#process.load("MuonAnalysis.MomentumScaleCalibration.Summer08_Upsilon1S_cff")

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
    "file:/home/demattia/Data/Z/Filter_Z_10.root",
    ),
    inputCommands = cms.untracked.vstring(
    "keep *",
    "drop *_TriggerResults_*_*")
)

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.CommonTopologies.globalTrackingGeometry_cfi")
process.load("RecoMuon.DetLayers.muonDetLayerGeometry_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("RecoMuon.TrackingTools.MuonServiceProxy_cff")
process.load("Alignment.CommonAlignmentProducer.GlobalPosition_Fake_cff")
process.load("MagneticField.Engine.uniformMagneticField_cfi")

# process.source = cms.Source("EmptySource")
# 
# process.maxEvents = cms.untracked.PSet(
#     input = cms.untracked.int32(0)
# )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)
process.looper = cms.Looper(
    "MuScleFit",
    process.MuonServiceProxy,

    # Choose the kind of muons you want to run on
    # -------------------------------------------

    # // all muons //
    MuonLabel = cms.InputTag("muons"),
    MuonType = cms.int32(1),
    # Defines what type of muons to use:
    # 0 = globalMuon
    # 1 = innerTrack
    # anything else = use all muons
    UseType = cms.untracked.uint32(1),

    # // standalone muons //
    # MuonLabel = cms.InputTag("standAloneMuons:UpdatedAtVtx"),
    # MuonType = cms.int32(2),

    # // tracker tracks //
    # MuonLabel = cms.InputTag("generalTracks"), # ctfWithMaterialTracks
    # MuonType = cms.int32(3),

    # // global muons (these are still reco::Tracks) //
    # MuonLabel = cms.InputTag("muons"),
    # MuonType = cms.int32(3),

    # Output settings
    # ---------------
    OutputFileName = cms.untracked.string('MuScleFit.root'),
    debug = cms.untracked.int32(0),

    # Likelihood settings
    # -------------------
    maxLoopNumber = cms.untracked.int32(3),
    # Select which fits to do in which loop (0 = do not, 1 = do)
    doResolFit =      cms.vint32(0, 1, 0),
    doScaleFit =      cms.vint32(1, 0, 0),
    doBackgroundFit = cms.vint32(0, 0, 0),

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
    # parBias = cms.vdouble(1.015, 0.001),

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
    #ResolFitType = cms.int32(8),
    #parResol = cms.vdouble(-0.003, 0.000205, 1.0, 0.445473,
    #                       0.00043, 0.0041, 0.0000028, 0.000077,
    #                       0.00011, 0.0018, -0.00000094, 0.000022),
    #parResolFix = cms.vint32(0, 0, 0, 0,
    #                         1, 1, 1, 1,
    #                         1, 1, 1, 1),
    #parResolOrder = cms.vint32(0, 0, 0, 0,
    #                           0, 0, 0, 0,
    #                           0, 0, 0, 0),

    # ------------------------------------------------- #
    # New resolution function derived for low Pt region #
    # ------------------------------------------------- #
    # The eleven parResol parameters of resolfittype=11 are respectively:
    #"offsetEtaCentral", "offsetEtaHigh", "coeffOverPt", "coeffHighPt", "linaerEtaCentral", "parabEtaCentral", "linaerEtaHigh", "parabEtaHigh" };
    ResolFitType = cms.int32(11), #inner tracks in 31X
    parResol = cms.vdouble(-0.986, -0.986, -0.04, -0.038, -0.0014, 0.006, -0.0025, 0.0185),
    parResolFix = cms.vint32(0, 0, 0, 0, 0, 0, 0, 0),
    parResolOrder = cms.vint32(0, 0, 0, 0, 0, 0, 0, 0),


    # -------------------- #
    # Scale fit parameters #
    # -------------------- #

    # Scale fit type=14: Pt offset and grade up to three, Eta terms up to the sixth grade
    # -----------------------------------------------------------------------------------
    ScaleFitType = cms.int32(14),
    parScaleOrder = cms.vint32(0,            # scale
                               0,0,0,        # pt up to grade 3
                               0,0,0,0,0,0), # eta up to grade 6
    parScaleFix =   cms.vint32(0,
                               0,0,0,
                               0,0,1,1,1,1),
    parScale = cms.vdouble(1.0,
                           -0.000000315315, 0., 0.,
                           0.0000147547, -0.00000836992, 0., 0., 0., 0.),

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
    LeftWindowFactor = cms.vdouble(1., 10., 10.),
    RightWindowFactor = cms.vdouble(1., 10., 10.),

    # The parameter of BgrFitType=1 is the bgr fraction
    # -------------------------------------------------
    # BgrFitType = cms.int32(1),
    # parBgrFix = cms.vint32(0),
    # parBgr = cms.vdouble(0.001),
    # parBgrOrder = cms.vint32(0),

    # The two parameters of BgrFitType=2 are respectively:
    # bgr fraction, (negative of) bgr exp. slope, bgr constant
    # --------------------------------------------------------
    # The function types for resonances in a region must be the same
    BgrFitType = cms.vint32(2, 2, 2), # regions
    # These empty parameters should be used when there is no background
    parBgr = cms.vdouble(0., 0.,   0., 0.,   0., 0.,
                         0., 0.,   0., 0.,   0., 0.,   0.,0.,   0.,0.,   0.,0.),
    parBgrFix = cms.vint32(0, 0,   0, 0,   0, 0,
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
    resfind = cms.vint32(1, 0, 0, 1, 0, 0),
    FitStrategy = cms.int32(2),

    speedup = cms.bool(False),
    OutputGenInfoFileName = cms.untracked.string("genSimRecoPlots.root"),
    # Set this to false if you do not want to use simTracks.
    # (Note that this is skipped anyway if speedup == True).
    compareToSimTracks = cms.bool(True),

    # This line is only necessary when running on fastSim
    # SimTracksCollection = cms.untracked.InputTag("fastSimProducer"),
    # This must be set to true when using events generated with Sherpa
    # Sherpa = cms.untracked.bool(True),

    # This line allows to switch to PAT muons. Default is false.
    # PATmuons = cms.untracked.bool(True),

    # This line allows to use the EDLooper or to loop by hand.
    # All the necessary information is saved during the first loop so there is not need
    # at this time to read again the events in successive iterations. Therefore by default
    # for iterations > 1 the loops are done by hand, which means that the framework does
    # not need to read all the events again. This is much faster.
    # If you need to read the events in every iteration put this to false.
    # FastLoop = cms.untracked.bool(False),


    # Only used when reading events from a root tree
    MaxEventsFromRootTree = cms.int32(-1)

    # Specify a file if you want to read events from a root tree in a local file.
    # In this case the input source should be an empty source with 0 events.
    InputRootTreeFileName = cms.string(""),
    # Specify the file name where you want to save a root tree with the muon pairs.
    # Leave empty if no file should be written.
    OutputRootTreeFileName = cms.string(""),

    # Fit accuracy and debug parameters
    StartWithSimplex = cms.bool(True),
    ComputeMinosErrors = cms.bool(False),
    MinimumShapePlots = cms.bool(False),

    TriggerResultsLabel = cms.untracked.InputTag("TEST"),
)

# Timing information
#process.load("FWCore.MessageLogger.MessageLogger_cfi")
#process.MessageLogger.logMuScleFit = cms.PSet(
process.MessageLogger = cms.Service(
    "MessageLogger",
    destinations = cms.untracked.vstring('logMuScleFit'),
    # logMuScleFit = cms.untracked.PSet(
    threshold = cms.untracked.string('INFO'),
    #default = cms.untracked.PSet(
    #    limit = cms.untracked.int32(10000000)
    #    )
    # )
)

# TimingLogFile = cms.untracked.string('timing.log')
# process.Timing = cms.Service("Timing")

process.load("MuonAnalysis.MomentumScaleCalibration.MuScleFitGenFilter_cfi")

path(MuScleFitGenFilter)
