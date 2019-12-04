import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
#process.load("MuonAnalysis.MomentumScaleCalibration.Summer08_Upsilon1S_cff")
#process.load("MuonAnalysis.MomentumScaleCalibration.Summer08_InclusivePPmuX_old_cff")
# process.load("MuonAnalysis.MomentumScaleCalibration.Summer09_Upsilon1S_prep_cff")
#process.load("MuonAnalysis.MomentumScaleCalibration.Summer08_Z_cff")

process.source = cms.Source(
    "PoolSource",
    inputCommands = cms.untracked.vstring(
    "keep *",
    "drop edmGenInfoProduct_*_*_*",
    "drop *_TriggerResults_*_*",
    ),
    fileNames = cms.untracked.vstring(
    #Aligned reco
    'rfio:/castor/cern.ch/user/m/maborgia/Z_CTEQ6l/Z0jetReco/FullReco0Jet_1-5000.root',
#     'rfio:/castor/cern.ch/user/m/maborgia/Z_CTEQ6l/Z0jetReco/FullReco0Jet_5001-10000.root',
#     'rfio:/castor/cern.ch/user/m/maborgia/Z_CTEQ6l/Z0jetReco/FullReco0Jet_10001-15000.root',
#     'rfio:/castor/cern.ch/user/m/maborgia/Z_CTEQ6l/Z0jetReco/FullReco0Jet_15001-20000.root',
#     'rfio:/castor/cern.ch/user/m/maborgia/Z_CTEQ6l/Z0jetReco/FullReco0Jet_20001-25000.root',
#     'rfio:/castor/cern.ch/user/m/maborgia/Z_CTEQ6l/Z0jetReco/FullReco0Jet_25001-30000.root',
#     'rfio:/castor/cern.ch/user/m/maborgia/Z_CTEQ6l/Z0jetReco/FullReco0Jet_30001-35000.root',
#     'rfio:/castor/cern.ch/user/m/maborgia/Z_CTEQ6l/Z0jetReco/FullReco0Jet_35001-40000.root',
#     'rfio:/castor/cern.ch/user/m/maborgia/Z_CTEQ6l/Z0jetReco/FullReco0Jet_40001-45000.root',
#     'rfio:/castor/cern.ch/user/m/maborgia/Z_CTEQ6l/Z0jetReco/FullReco0Jet_45001-50000.root'
    #Misaligned reco
#    'rfio:/castor/cern.ch/user/m/maborgia/Z_CTEQ6l/Z0jetReco/ZMuMu0jet_ReRecoWMisalignCRAFT.root'    
    )
)


process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.CommonTopologies.globalTrackingGeometry_cfi")
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

    input = cms.untracked.int32(5000)
)
process.looper = cms.Looper(
    "MuScleFit",
    process.MuonServiceProxy,
    ProbabilitiesFile = cms.untracked.string('/tmp/maborgia/Probs_new_Horace_CTEQ_1000.root'),
    # Choose the kind of muons you want to run on
    # -------------------------------------------

    # // global muons //
    # MuonLabel = cms.InputTag("muons"),
    # MuonType = cms.int32(1),

    # // inner track //
    MuonLabel = cms.InputTag("muons"),
    MuonType = cms.int32(10),

    MaxMuonPt = cms.untracked.double(50.),
    MinMuonPt = cms.untracked.double(20.),
    
    # // standalone muons //
    # MuonLabel = cms.InputTag("standAloneMuons:UpdatedAtVtx"),
    # muonType = cms.int32(2),

    # // tracker tracks //
    # MuonLabel = cms.InputTag("generalTracks"), # ctfWithMaterialTracks
    # muonType = cms.int32(3),

    # Output settings
    # ---------------
    OutputFileName = cms.untracked.string('MuScleFit.root'),
    debug = cms.untracked.int32(0),

    # Likelihood settings
    # -------------------
    maxLoopNumber = cms.untracked.int32(3),
    # Select which fits to do in which loop (0 = do not, 1 = do)
    doResolFit =      cms.vint32( 1, 0, 0),
    doScaleFit =      cms.vint32( 0, 1, 0),
    doBackgroundFit = cms.vint32( 0, 0, 0),

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
#     SmearType = cms.int32(0),
#     parSmear = cms.vdouble(),

    # =======> To use with resolution function type 15
    SmearType = cms.int32(6),
    parSmear = cms.vdouble(-0.00027357, 0., 0.000319814,
                           0., 0., 0.00508154,
                           0., 0.,  1.18468, 0.0756988,
                           0., -0.0349113, 1.13643, 0.0719739,
                           1.0008, 0.050805),
    
    # ------------------------- #
    # Resolution fit parameters #
    # ------------------------- #

    # ------------------------------------------------- #
    # New resolution function derived for low Pt region #
    # ------------------------------------------------- #
    # The eleven parResol parameters of resolfittype=11 are respectively:
    #"offsetEtaCentral", "offsetEtaHigh", "coeffOverPt", "coeffHighPt", "linaerEtaCentral", "parabEtaCentral", "linaerEtaHigh", "parabEtaHigh" };
    #ResolFitType = cms.int32(11), #inner tracks in 31X
    #parResol = cms.vdouble(-0.986, -0.986, -0.04, -0.038, -0.0014, 0.006, -0.0025, 0.0185),
    #parResolFix = cms.vint32(0, 0, 0, 0, 0, 0, 0, 0),
    #parResolOrder = cms.vint32(0, 0, 0, 0, 0, 0, 0, 0),

    # ResolFitType = cms.int32(12), #inner tracks in 31X
    # parResol = cms.vdouble(-0.986, -0.986, -0.04, -0.038,
    #                        -0.0014, 0.006, -0.0025, 0.0185,
    #                        -0.0014, 0., -0.001),
    # parResolFix = cms.vint32(0, 0, 0, 0,
    #                          0, 0, 0, 0,
    #                          0, 0, 0),
    # parResolOrder = cms.vint32(0, 0, 0, 0,
    #                            0, 0, 0, 0,
    #                            0, 0, 0),

#     ResolFitType = cms.int32(8),
#     parResol = cms.vdouble(0.00, 0.0, 1.38, 0.051,
#                            0.00043, 0.0041, 0.0000028, 0.000077,
#                            0.00011, 0.0018, -0.00000094, 0.000022),
#     parResolFix = cms.vint32(1, 1, 0, 0,
#                              1, 1, 1, 1,
#                              1, 1, 1, 1),
#     parResolOrder = cms.vint32(0, 0, 0, 0,
#                                0, 0, 0, 0,
#                                0, 0, 0, 0),

#### Resolution function for misaligned data Type 15
    ResolFitType = cms.int32(15),
    parResol = cms.vdouble(0.02, 0.0, 0.00014, 
                           0.0, 0.0, 0.005759,
                           0.0, 0.0, 1.38, 0.114,
                           0.0, 0.0, 1.4856, 0.0954
                           ),
    parResolFix = cms.vint32(0, 1, 0,
                             1, 1, 0,
                             1, 1, 0, 0,
                             1, 0, 0, 0),
    parResolOrder = cms.vint32(0, 0, 0,
                               0, 0, 0,
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
                               0,1,1,
                               1,1,1,1,1,1),
    parScale = cms.vdouble(1.0,
                           0., 0., 0.,
                           0., 0., 0., 0., 0., 0.),

    #     parScale = cms.vdouble(1.0,
#                            -0.000000315315, 0., 0.,
#                            0.0000147547, -0.00000836992, 0., 0., 0., 0.),
#     parScale = cms.vdouble(1.0,
#                             -0.00041991, 0., 0.,
#                            0.000727967, -0.00082597, 0., 0., 0., 0.),

    # ------------------------- #
    # Background fit parameters #
    # ------------------------- #

    # Window factors for: Z, Upsilons and (J/Psi,Psi2S) regions
    LeftWindowFactor = cms.vdouble(1., 5., 3.),
    RightWindowFactor = cms.vdouble(1., 5., 3.),

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
    BgrFitType = cms.vint32(2, 2, 2), # resonances
    # parBgr = cms.vdouble(0.05, 0.001),
    parBgr = cms.vdouble(0., 0.,   0., 0.,   0., 0.,
                         0., 0.,   0., 0.,   0., 0.,   0.,0.,   0.,0.,   0.,0.),
    # parBgr = cms.vdouble(0., 0.,   0.9, 0.0001,   0., 0.,
    #                      0., 0.,   0., 0.,   0., 0.,   0.,0.,   0.,0.,   0.,0.),
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
    resfind = cms.vint32(1, 0, 0, 0, 0, 0),
    FitStrategy = cms.int32(2),

    speedup = cms.bool(False),
    # speedup = cms.bool(True),
    # Set this to false if you do not want to use simTracks.
    # (Note that this is skipped anyway if speedup == True).
    # compareToSimTracks = cms.bool(True),
    compareToSimTracks = cms.bool(False),
    Sherpa = cms.untracked.bool(True),

)

# Timing information
# process.load("FWCore.MessageLogger.MessageLogger_cfi")
# TimingLogFile = cms.untracked.string('timing.log')
# process.Timing = cms.Service("Timing")


