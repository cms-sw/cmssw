import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("MuonAnalysis.MomentumScaleCalibration.local_CSA08_JPsi_cff")

# Conflicts with Uniform magnetic field, because it delivers VolumeBasedMagneticField
# process.load("Configuration.StandardSequences.MagneticField_cff")

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

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.poolDBESSource = cms.ESSource("PoolDBESSource",
   BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
   DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(2),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:/data2/demattia/CMSSW_2_1_12/src/MuonAnalysis/MomentumScaleCalibration/test/dummy2.db'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('MuScleFitLikelihoodPdfRcd'),
        tag = cms.string('MuScleFitLikelihoodPdf_2_1_12')
    ))
)

process.looper = cms.Looper(
    "MuScleFit",
    process.MuonServiceProxy,

    # Choose the kind of muons you want to run on
    # -------------------------------------------

    # // global muons //
    # MuonLabel = cms.InputTag("muons"),
    # muonType = cms.int32(1),

    # // standalone muons //
    # MuonLabel = cms.InputTag("standAloneMuons:UpdatedAtVtx"),
    # muonType = cms.int32(2),

    # // tracker tracks //
    MuonLabel = cms.InputTag("generalTracks"), # ctfWithMaterialTracks
    muonType = cms.int32(3),

    # Output settings
    # ---------------
    RootFileName = cms.untracked.string('MuScleFit.root'),
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
    # parBias = cms.vdouble(1.015, 0.0),

    # Sinusoidal in phi
    # -----------------
    # BiasType = 2 means sinusoidal bias on the muons Pt
    # the two parameters are defined by:
    # pt = (parScale[0] + parScale[1]*sin(phi))*pt; 
    # BiasType = cms.int32(2),
    # parBias = cms.vdouble(1.015, 0.05),

    # The 8 parameters of BiasType=13 are respectively:
    # Pt scale, Pt slope, Eta slope, Eta quadr., 
    # phi ampl. mu+, phi phase mu+, phi ampl. mu-, phi phase mu-
    # ----------------------------------------------------------
    # BiasType = cms.int32(13),
    # parBias = cms.vdouble(1., 0., 0., 0., 0.01, 1., 0.01, 1.5),

    # SmearType=0 means no smearing applied to muon momenta
    # -----------------------------------------------------
    SmearType = cms.int32(0),
    parSmear = cms.vdouble(),

    # The six parameters of SmearType=4 are respectively:
    # Pt dep. of Pt res., |eta| dep. of Pt res., Phi res., |eta| res., 
    # |eta| dep. of |eta| res., Pt^2 dep. of Pt res.
    # ----------------------------------------------------------------
    # SmearType = cms.int32(4),
    # parSmear = cms.vdouble(0., 0., 0., 0., 0., 0.), 

    # ------------------------- #
    # Resolution fit parameters #
    # ------------------------- #

    # The three parameters of resolfittype=1:
    # pt, phi and cotgth.
    # ---------------------------------------------------------------
    # ResolFitType = cms.int32(1),
    # parResol = cms.vdouble(0.01, 0.01, 0.01),
    # parResolFix = cms.vint32(0, 0, 0),
    # parResolOrder = cms.vint32(0, 0, 0),

    # The four parameters of resolfittype=2 are respectively:
    # Pt dep. of Pt res., |eta| dep. of Pt res., Phi res., |eta| res.
    # ---------------------------------------------------------------
    # ResolFitType = cms.int32(2),
    # parResol = cms.vdouble(0.0076, 0.061, 0.0051, 0.0025),
    # parResolFix = cms.vint32(0, 0, 0, 0),
    # parResolOrder = cms.vint32(0, 0, 0, 0),

    # The five parameters of resolfittype=3 are respectively:
    # Pt dep. of Pt res., |eta| dep. of Pt res., Phi res., |eta| res.
    # |eta| dep of eta res.
    # ---------------------------------------------------------------
    # ResolFitType = cms.int32(2),
    # parResol = cms.vdouble(0.0076, 0.061, 0.0051, 0.0025, 0.001),
    # parResolFix = cms.vint32(0, 0, 0, 0, 0),
    # parResolOrder = cms.vint32(0, 0, 0, 0, 0),

    # The six parameters of resolfittype=4 are respectively:
    # Pt dep. of Pt res., |eta| dep. of Pt res., Phi res., |eta| res., 
    # |eta| dep of eta res., Pt^2 dep. of Pt res.
    # ----------------------------------------------------------------
    # ResolFitType = cms.int32(4),        
    # parResol = cms.vdouble(0.001, 0.001, 0.001, 0.001, 0.001, 0.0000001),
    # parResolFix = cms.vint32(0, 0, 0, 0, 0, 0), 
    # parResolOrder = cms.vint32(0, 0, 0, 0, 0, 0),

    # The seven parResol parameters of resolfittype=5 are respectively:
    # Pt dep. of Pt res., |eta| dep. of Pt res., Phi res., |eta| res., 
    # Pt^2 dep. of Pt res., costant in eta res., phi dep. of phi res.
    # ----------------------------------------------------------------
    # ResolFitType = cms.int32(5),
    # parResol = cms.vdouble(0.0085, 0.049, 0.0008, 0.0, 0.0067, 1e-05, 0.0002),
    # parResolFix = cms.vint32(0, 0, 0, 0, 0, 0, 0),
    # parResolOrder = cms.vint32(0, 0, 0, 0, 0, 0, 0),

    # The fifteen parResol parameters of resolfittype=6 are respectively:
    # constant of sigmaPt, Pt dep. of sigmaPt, Pt^2 dep. of sigmaPt;
    # Pt^3 dep. of sigmaPt and Pt^4 dep. of sigmaPt;
    # constant of sigmaCotgTheta, 1/Pt dep. of sigmaCotgTheta, Eta dep. of
    # sigmaCotgTheta, Eta^2 dep of sigmaCotgTheta;
    # constant of sigmaPhi, 1/Pt dep. of sigmaPhi, Eta dep. of sigmaPhi,
    # Eta^2 dep. of sigmaPhi.
    # ----------------------------------------------------------------
    # ResolFitType = cms.int32(6),
    # parResol = cms.vdouble(0.002, -0.0015, 0.000056, -0.00000085, 0.0000000046, -0.000027, 0.0037,
    #                        0.005, 0.00027, 0.0000027, 0.000094,
    #                        0.002, 0.00016, -0.00000051, 0.000022),
    # parResolFix = cms.vint32(0, 0, 0, 0, 0, 0, 0, 0,
    #                          0, 0, 0, 0, 0, 0, 0),
    # parResolOrder = cms.vint32(0, 0, 0, 0, 0, 0, 0, 0,
    #                            0, 0, 0, 0, 0, 0, 0),

    # The twelve parResol parameters of resolfittype=7 are respectively:
    # constant of sigmaPt, Pt dep. of sigmaPt;
    # eta and eta^2 dep. of sigmaPt.
    # constant of sigmaCotgTheta, 1/Pt dep. of sigmaCotgTheta, Eta dep. of
    # sigmaCotgTheta, Eta^2 dep of sigmaCotgTheta;
    # constant of sigmaPhi, 1/Pt dep. of sigmaPhi, Eta dep. of sigmaPhi,
    # Eta^2 dep. of sigmaPhi.
    # ----------------------------------------------------------------
    # ResolFitType = cms.int32(7),
    # parResol = cms.vdouble(0.012, 0.0001, 0.000019, 0.0027,
    #                        0.00043, 0.0041, 0.0000028, 0.000077,
    #                        0.00011, 0.0018, -0.00000094, 0.000022),
    # parResolFix = cms.vint32(0, 0, 0, 0,
    #                          0, 0, 0, 0,
    #                          0, 0, 0, 0),
    # parResolOrder = cms.vint32(0, 0, 0, 0,
    #                            1, 1, 1, 1,
    #                            2, 2, 2, 2),

    # The eleven parResol parameters of resolfittype=8 are respectively:
    # constant of sigmaPt, Pt dep. of sigmaPt,
    # scale of the eta dep. made by points with values derived from MuonGun.
    # constant of sigmaCotgTheta, 1/Pt dep. of sigmaCotgTheta, Eta dep. of
    # sigmaCotgTheta, Eta^2 dep of sigmaCotgTheta;
    # constant of sigmaPhi, 1/Pt dep. of sigmaPhi, Eta dep. of sigmaPhi,
    # Eta^2 dep. of sigmaPhi.
    # ----------------------------------------------------------------
    ResolFitType = cms.int32(8),
    parResol = cms.vdouble(-0.003, 0.000205, 1.0,
                           0.00043, 0.0041, 0.0000028, 0.000077,
                           0.00011, 0.0018, -0.00000094, 0.000022),
    parResolFix = cms.vint32(0, 0, 0,
                             1, 1, 1, 1,
                             1, 1, 1, 1),
    parResolOrder = cms.vint32(0, 0, 0,
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

    # Scale fit type=8: Pt offset and slope, Eta slope and quadratic term
    # -------------------------------------------------------------------
    ScaleFitType = cms.int32(8),
    parScaleOrder = cms.vint32(0,0,0,0),
    parScaleFix = cms.vint32(0,0,0,0),
    parScale = cms.vdouble(1.0, -0.000000315315,0.0000147547,-0.00000836992),

    # The 8 parameters of ScaleFitType=13 are respectively:
    # Pt scale, Pt slope, Eta slope, Eta quadr., 
    # phi ampl. mu+, phi phase mu+, phi ampl. mu-, phi phase mu-
    # ----------------------------------------------------------
    # ScaleFitType = cms.int32(13),
    # parScaleOrder = cms.vint32(0, 0, 0, 0, 0, 0, 0, 0),
    # parScaleFix = cms.vint32(0, 1, 1, 1, 0, 0, 0, 0),
    # parScale = cms.vdouble(1.0, 0.0, 0.0, 0.0, 0.0, 1.57, 0.0, 1.57),

    # ------------------------- #
    # Background fit parameters #
    # ------------------------- #

    # The parameter of BgrFitType=1 is the bgr fraction
    # -------------------------------------------------
    BgrFitType = cms.int32(1),
    parBgrFix = cms.vint32(0),
    # parBgr = cms.vdouble(0.05),
    # No background for now
    parBgr = cms.vdouble(0.0),
    parBgrOrder = cms.vint32(0),

    # The two parameters of BgrFitType=2 are respectively:
    # bgr fraction, (negative of) bgr exp. slope, bgr constant
    # --------------------------------------------------------
    # BgrFitType = cms.int32(2),
    # parBgr = cms.vdouble(0.05, 0.001),
    # parBgrFix = cms.vint32(0, 0),
    # parBgrOrder = cms.vint32(0, 0),

    # The three parameters of BgrFitType=3 are respectively:
    # bgr fraction, (negative of) bgr exp. slope, bgr constant
    # --------------------------------------------------------
    # BgrFitType = cms.vint32(3),
    # parBgr = cms.vdouble(0.05, 0.001, 0.001),
    # parBgrFix = cms.vint32(0, 0, 0),
    # parBgrOrder = cms.vint32(0, 0, 0),

    # ---------------- #
    # Select resonance #
    # ---------------- #

    # The resonances are to be specified in this order:
    # Z0, Y(3S), Y(2S), Y(1S), Psi(2S), J/Psi
    # -------------------------------------------------
    resfind = cms.vint32(0, 0, 0, 0, 0, 1),
    FitStrategy = cms.int32(2),

    speedup = cms.bool(False),
    # readPdfFromDB = cms.bool(False)

)

# Timing information
process.load("FWCore.MessageLogger.MessageLogger_cfi")
TimingLogFile = cms.untracked.string('timing.log')
process.Timing = cms.Service("Timing")


