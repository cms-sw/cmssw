import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("MuonAnalysis.MomentumScaleCalibration.local_CSA08_JPsi_cff")

process.load("Configuration.StandardSequences.MagneticField_cff")

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
process.looper = cms.Looper(
    "MuScleFit",
    process.MuonServiceProxy,

    # Choose the kind of muons you want to run on
    # -------------------------------------------

    # // global muons //
    MuonLabel = cms.InputTag("muons"),
    muonType = cms.int32(1),

    # // standalone muons //
    # MuonLabel = cms.InputTag("standAloneMuons:UpdatedAtVtx")
    # muonType = cms.int32(2), 

    # // tracker tracks //
    # MuonLabel = cms.InputTag("generalTracks") //ctfWithMaterialTracks
    # muonType = cms.int32(3), 

    # Output settings
    # ---------------
    RootFileName = cms.untracked.string('MuScleFit.root'),
    debug = cms.untracked.int32(0),

    # Likelihood settings
    # -------------------
    maxLoopNumber = cms.untracked.int32(5),

    # Fit parameters and fix flags (1 = use par)
    # ==========================================

    # The 8 parameters of BiasType=13 are respectively:
    # Pt scale, Pt slope, Eta slope, Eta quadr., 
    # phi ampl. mu+, phi phase mu+, phi ampl. mu-, phi phase mu-
    # ----------------------------------------------------------
    #BiasType = cms.int32(13),
    #parBias = cms.vdouble(1., 0., 0., 0., 0.01, 1., 0.01, 1.5),

    # BiasType=0 means no bias to muon momenta
    # ----------------------------------------
    BiasType = cms.int32(0),
    parBias = cms.vdouble(),

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

    # The seven parResol parameters of resolfittype=5 are respectively:
    # Pt dep. of Pt res., |eta| dep. of Pt res., Phi res., |eta| res., 
    # Pt^2 dep. of Pt res., costant in eta res., phi dep. of phi res.
    # ----------------------------------------------------------------
    ResolFitType = cms.int32(5),

    parResol = cms.vdouble(0.0085, 0.049, 0.0008, 0.0, 0.0067, 
        1e-05, 0.0002),
    parResolFix = cms.vint32(0, 0, 0, 0, 0, 0, 0),
    parResolOrder = cms.vint32(0, 0, 0, 0, 0, 0, 0),

    # The four parameters of resolfittype=2 are respectively:
    # Pt dep. of Pt res., |eta| dep. of Pt res., Phi res., |eta| res.
    # ---------------------------------------------------------------
    # ResolFitType = cms.int32(2),
    # parResol = cms.vdouble(0.0076, 0.061, 0.0051, 0.0025),
    # parResolFix = cms.vint32(0, 0, 0, 0), 
    # parResolOrder = cms.vint32(0, 0, 0, 0),

    # The six parameters of resolfittype=4 are respectively:
    # Pt dep. of Pt res., |eta| dep. of Pt res., Phi res., |eta| res., 
    # |eta| dep of eta res., Pt^2 dep. of Pt res.
    # ----------------------------------------------------------------
    # ResolFitType = cms.int32(4),        
    # parResol = cms.vdouble(0.001, 0.001, 0.001, 0.001, 0.001, 0.0000001),
    # parResolFix = cms.vint32(0, 0, 0, 0, 0, 0), 
    # parResolOrder = cms.vint32(0, 0, 0, 0, 0, 0),

    # The 8 parameters of ScaleFitType=13 are respectively:
    # Pt scale, Pt slope, Eta slope, Eta quadr., 
    # phi ampl. mu+, phi phase mu+, phi ampl. mu-, phi phase mu-
    # ----------------------------------------------------------
    ScaleFitType = cms.int32(13),
    parScaleOrder = cms.vint32(0, 0, 0, 0, 0, 
        0, 0, 0),

    parScaleFix = cms.vint32(0, 1, 1, 1, 0, 0, 0, 0),
    parScale = cms.vdouble(1.0, 0.0, 0.0, 0.0, 0.0, 
                           1.57, 0.0, 1.57),

    # The parameter of BgrFitType=1 is the bgr fraction
    # -------------------------------------------------
    BgrFitType = cms.int32(1),
    parBgrFix = cms.vint32(0),
    parBgr = cms.vdouble(0.05),
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

    # The resonances are to be specified in this order:
    # Z0, Y(3S), Y(2S), Y(1S), Psi(2S), J/Psi
    # -------------------------------------------------
    resfind = cms.vint32(0, 0, 0, 0, 0, 1),
    FitStrategy = cms.int32(2),

    speedup = cms.bool(False)

)


