# -*- coding: utf-8 -*-
import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST2")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.CommonTopologies.globalTrackingGeometry_cfi")
process.load("RecoMuon.DetLayers.muonDetLayerGeometry_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("RecoMuon.TrackingTools.MuonServiceProxy_cff")
process.load("Alignment.CommonAlignmentProducer.GlobalPosition_Fake_cff")
process.load("MagneticField.Engine.uniformMagneticField_cfi")

process.source = cms.Source(
    "PoolSource",
    inputCommands = cms.untracked.vstring(
    "keep *"
    ),
    fileNames = cms.untracked.vstring(
    "INPUTFILENAME"
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# process.source = cms.Source("EmptySource")

# process.maxEvents = cms.untracked.PSet(
#     input = cms.untracked.int32(0)
# )

process.looper = cms.Looper(
    "MuScleFit",
    process.MuonServiceProxy,

    # Choose the kind of muons you want to run on
    # -------------------------------------------

    # // all muons //
    MuonLabel = cms.InputTag("patMuons"),
    # Defines what type of muons to use:
    # -1 = onia guys selection
    # -2 = onia guys selection - only GG
    # -3 = onia guys selection - only GT
    # -4 = onia guys selection - only TT
    # Note that the above samples are independent and represent the composition of the inclusive sample
    # 0 = muon (use the values in the muon collection)
    # 1 = global muon
    # 2 = standalone muon
    # 3 = tracker muon
    # 4 = calo muon
    # 10 = innerTrack of muon
    # 11 = innerTrack of global muon
    # 13 = innerTrack of tracker muon
    MuonType = cms.int32(-1),

    # Output settings
    # ---------------
    OutputFileName = cms.untracked.string('MuScleFit.root'),
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

    ResolFitType = cms.int32(12),
    parResol = cms.vdouble(1.33, -0.990, 0.00762, -0.00341,
                           0.00480, 0.00223, 0.00560, 0.0118,
                           1.61, 0.000322, 0.000004,
                           -0.00252, 0.00863, 0.00745, -0.00337),
    parResolFix = cms.vint32(0, 0, 0, 0,
                             0, 0, 0, 0,
                             0, 0, 0,
                             0, 0, 0, 0),
    parResolOrder = cms.vint32(0, 0, 0, 0,
                               0, 0, 0, 0,
                               0, 0, 0,
                               0, 0, 0, 0),

    # -------------------- #
    # Scale fit parameters #
    # -------------------- #
    # Scale fit type=8: linear in pt and parabolic in eta with four parameters
    # ------------------------------------------------------------------------
    # ScaleFitType = cms.int32(8),
    # parScaleOrder = cms.vint32(0,0,0,0),
    # parScaleFix =   cms.vint32(0,0,0,0),
    # parScale = cms.vdouble(1.001, 0., 0., 0.),

    ScaleFitType = cms.int32(14),
    parScaleOrder = cms.vint32(0,0,0,0,
                               0,0,0,0,0,0),
    parScaleFix =   cms.vint32(0,1,1,1,
                               0,0,0,1,1,1),
    parScale = cms.vdouble(1., 0., 0., 0.,
                           0., 0., 0., 0., 0., 0.),

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
    LeftWindowBorder = cms.vdouble(60., 8., 1.391495),
    RightWindowBorder = cms.vdouble(120., 12., 5.391495),

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
    parBgr = cms.vdouble(0., 0.,   0., 0.,   0., 0.,
                         0., 0.,   0., 0.,   0., 0.,     0.,0.,   0.,0.,   0.,0.),
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
    resfind = cms.vint32(1, 1, 1, 1, 1, 1),
    FitStrategy = cms.int32(2),

    speedup = cms.bool(False),
    # Set this to false if you do not want to use simTracks.
    # (Note that this is skipped anyway if speedup == True).
    compareToSimTracks = cms.bool(False),
    Sherpa = cms.untracked.bool(False),
    DebugMassResol = cms.untracked.bool(False),

    PATmuons = cms.untracked.bool(True),
    genParticles = cms.bool(True),
    GenParticlesName = cms.untracked.string('genParticles'),
    HepMCProduct = cms.bool(False),

    UseProbsFile = cms.untracked.bool(False),
    ProbabilitiesFileInPath = cms.untracked.string(""),

    # Only used when reading events from a root tree
    MaxEventsFromRootTree = cms.int32(-1),

    InputRootTreeFileName = cms.string(""),
    OutputRootTreeFileName = cms.string("OUTPUTTREENAME"),

    # Fit accuracy and debug parameters
    StartWithSimplex = cms.bool(True),
    ComputeMinosErrors = cms.bool(False),
    MinimumShapePlots = cms.bool(False),

    # The following parameters can be used to filter events
    TriggerResultsLabel = cms.untracked.string(""),
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


