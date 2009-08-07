import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
#process.load("MuonAnalysis.MomentumScaleCalibration.Summer08_Upsilon1S_cff")

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
    "file:/data2/demattia/Data/Z/Filter_Z_10.root",
    "file:/data2/demattia/Data/Z/Filter_Z_11.root",
    "file:/data2/demattia/Data/Z/Filter_Z_12.root",
    "file:/data2/demattia/Data/Z/Filter_Z_13.root",
    "file:/data2/demattia/Data/Z/Filter_Z_14.root",
    "file:/data2/demattia/Data/Z/Filter_Z_15.root",
    "file:/data2/demattia/Data/Z/Filter_Z_16.root",
    "file:/data2/demattia/Data/Z/Filter_Z_17.root",
    "file:/data2/demattia/Data/Z/Filter_Z_18.root",
    "file:/data2/demattia/Data/Z/Filter_Z_19.root",
    "file:/data2/demattia/Data/Z/Filter_Z_1.root",
    "file:/data2/demattia/Data/Z/Filter_Z_20.root",
    "file:/data2/demattia/Data/Z/Filter_Z_21.root",
    "file:/data2/demattia/Data/Z/Filter_Z_22.root",
    "file:/data2/demattia/Data/Z/Filter_Z_23.root",
    "file:/data2/demattia/Data/Z/Filter_Z_24.root",
    "file:/data2/demattia/Data/Z/Filter_Z_25.root",
    "file:/data2/demattia/Data/Z/Filter_Z_26.root",
    "file:/data2/demattia/Data/Z/Filter_Z_27.root",
    "file:/data2/demattia/Data/Z/Filter_Z_28.root",
    "file:/data2/demattia/Data/Z/Filter_Z_29.root",
    "file:/data2/demattia/Data/Z/Filter_Z_2.root",
    "file:/data2/demattia/Data/Z/Filter_Z_30.root",
    "file:/data2/demattia/Data/Z/Filter_Z_31.root",
    "file:/data2/demattia/Data/Z/Filter_Z_32.root",
    "file:/data2/demattia/Data/Z/Filter_Z_33.root",
    "file:/data2/demattia/Data/Z/Filter_Z_34.root",
    "file:/data2/demattia/Data/Z/Filter_Z_35.root",
    "file:/data2/demattia/Data/Z/Filter_Z_36.root",
    "file:/data2/demattia/Data/Z/Filter_Z_37.root",
    "file:/data2/demattia/Data/Z/Filter_Z_38.root",
    "file:/data2/demattia/Data/Z/Filter_Z_39.root",
    "file:/data2/demattia/Data/Z/Filter_Z_3.root",
    "file:/data2/demattia/Data/Z/Filter_Z_40.root",
    "file:/data2/demattia/Data/Z/Filter_Z_41.root",
    "file:/data2/demattia/Data/Z/Filter_Z_42.root",
    "file:/data2/demattia/Data/Z/Filter_Z_43.root",
    "file:/data2/demattia/Data/Z/Filter_Z_44.root",
    "file:/data2/demattia/Data/Z/Filter_Z_45.root",
    "file:/data2/demattia/Data/Z/Filter_Z_46.root",
    "file:/data2/demattia/Data/Z/Filter_Z_47.root",
    "file:/data2/demattia/Data/Z/Filter_Z_48.root",
    "file:/data2/demattia/Data/Z/Filter_Z_49.root",
    "file:/data2/demattia/Data/Z/Filter_Z_4.root",
    "file:/data2/demattia/Data/Z/Filter_Z_50.root",
    "file:/data2/demattia/Data/Z/Filter_Z_5.root",
    "file:/data2/demattia/Data/Z/Filter_Z_6.root",
    "file:/data2/demattia/Data/Z/Filter_Z_7.root",
    "file:/data2/demattia/Data/Z/Filter_Z_8.root",
    "file:/data2/demattia/Data/Z/Filter_Z_9.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_10.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_11.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_12.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_13.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_14.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_15.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_16.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_17.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_18.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_19.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_1.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_20.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_21.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_22.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_23.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_24.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_25.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_26.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_27.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_28.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_29.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_2.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_30.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_31.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_32.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_33.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_34.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_35.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_36.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_37.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_38.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_39.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_3.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_40.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_41.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_42.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_43.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_44.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_45.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_46.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_47.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_48.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_49.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_4.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_50.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_5.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_6.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_7.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_8.root",
    "file:/data2/demattia/Data/Upsilon/Filter_Upsilon_9.root"
    ),
    inputCommands = cms.untracked.vstring(
    "keep *",
    "drop *_TriggerResults_*_*")
)

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.CommonDetUnit.globalTrackingGeometry_cfi")
process.load("RecoMuon.DetLayers.muonDetLayerGeometry_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("RecoMuon.TrackingTools.MuonServiceProxy_cff")
process.load("Alignment.CommonAlignmentProducer.GlobalPosition_Fake_cff")
process.load("MagneticField.Engine.uniformMagneticField_cfi")

# process.source = cms.Source("PoolSource",
#     fileNames = cms.untracked.vstring()
# )

#process.poolDBESSource = cms.ESSource("PoolDBESSource",
#   BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
#   DBParameters = cms.PSet(
#        messageLevel = cms.untracked.int32(2),
#        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
#    ),
#    timetype = cms.untracked.string('runnumber'),
#    connect = cms.string('sqlite_file:/data2/demattia/CMSSW_2_1_12/src/MuonAnalysis/MomentumScaleCalibration/test/dummy2.db'),
#    toGet = cms.VPSet(cms.PSet(
#        record = cms.string('MuScleFitLikelihoodPdfRcd'),
#        tag = cms.string('MuScleFitLikelihoodPdf_2_1_12')
#    ))
#)

process.maxEvents = cms.untracked.PSet(
    # This are the total background events from InclusivePPmuX (89150) +
    # the number of Upsilon1S events.
    # input = cms.untracked.int32(89355)

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

    # The eleven parResol parameters of resolfittype=8 are respectively:
    # constant of sigmaPt, Pt dep. of sigmaPt, Pt^2 dep. of sigmaPt,
    # scale of the eta dep. made by points with values derived from MuonGun,
    # border value (higher eta) of etaByPoints;
    # constant of sigmaCotgTheta, 1/Pt dep. of sigmaCotgTheta, Eta dep. of
    # sigmaCotgTheta, Eta^2 dep of sigmaCotgTheta;
    # onstant of sigmaPhi, 1/Pt dep. of sigmaPhi, Eta dep. of sigmaPhi,
    # Eta^2 dep. of sigmaPhi.
    # ----------------------------------------------------------------
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

    ResolFitType = cms.int32(9),
    parResol = cms.vdouble(0.00118247, 0.000365709, -0.00000181061, -0.000000207808, 0.00000000143758, 0.600788, 0.0416943,
                           0.00043, 0.0041, 0.0000028, 0.000077,
                           0.00011, 0.0018, -0.00000094, 0.000022,
                           0.00496148, 0.00496148, 0.00448753, 0.00448753,
                           0.00468222, 0.00468222, 0.00392335, 0.00392335,
                           0.296549, 0.296549, 0.360698, 0.360698,
                           0.452533, 0.452533, 0.419223, 0.419223),
    parResolFix = cms.vint32(0, 0, 0, 0, 0, 0, 0,
                             1, 1, 1, 1,
                             1, 1, 1, 1,
                             0, 0, 0, 0,
                             0, 0, 0, 0,
                             0, 0, 0, 0,
                             0, 0, 0, 0),
    parResolOrder = cms.vint32(0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0,
                               0, 0, 0, 0,
                               0, 0, 0, 0,
                               0, 0, 0, 0,
                               0, 0, 0, 0,
                               0, 0, 0, 0),

    # -------------------- #
    # Scale fit parameters #
    # -------------------- #

    # Fit a linear Pt scale correction with parameters:
    # Pt scale and Pt slope.
    # -------------------------------------------------
    # ScaleFitType = cms.int32(1),
    # parScaleOrder = cms.vint32(0,0),
    # parScaleFix = cms.vint32(0,0),
    # parScale = cms.vdouble(1.0, 0.0),

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
    # SimTracksCollection = cms.untracked.InputTag("famosSimHits"),
    # This must be set to true when using events generated with Sherpa
    # Sherpa = cms.untracked.bool(True),
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
