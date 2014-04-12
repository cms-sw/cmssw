import FWCore.ParameterSet.Config as cms

class SetParameters:
    def __init__(self):
        self.parResolFix = cms.vint32()
        self.parResolOrder = cms.vint32()
        self.parResol = cms.vdouble()
        self.parResolStep = cms.untracked.vdouble()
        self.parResolMin  = cms.untracked.vdouble()
        self.parResolMax  = cms.untracked.vdouble()

    def set(self, fix, order, value, step, min, max):
        self.parResolFix.append(fix)
        self.parResolOrder.append(order)
        self.parResol.append(value)
        self.parResolStep.append(step)
        self.parResolMin.append(min)
        self.parResolMax.append(max)

setter = SetParameters()
#fix, order, value, step, min, max
setter.set( 0 ,0, -0.00113112 , 0.002,      -0.1,  0.1  )
setter.set( 0 ,0, 0.000246547 , 0.00002,    -0.01, 0.01 )
setter.set( 0 ,0, 0.00545563  , 0.000002,   0.,    0.01 )
setter.set( 0 ,0, 0.00501745  , 0.0002,     -0.01, 0.02 )
setter.set( 1 ,0, 0.0091978   , 0.00002,    0.,    0.01 )
setter.set( 0 ,0, 0.0999428   , 0.0002,     0.,    0.2  )
setter.set( 0 ,0, 0.0484629   , 0.0000002,  -0.2,  0.5  )
setter.set( 0 ,0, -1.24738    , 0.0002,     -2.2,  -0.8 )
setter.set( 1 ,0, 0.          , 0.00002,    0.,    0.01 )
setter.set( 0 ,0, -0.0499885  , 0.0002,     -0.2,  0.1  )
setter.set( 0 ,0, 0.252381    , 0.000002,   -0.1,  0.5  )
setter.set( 0 ,0, 1.75024     , 0.0002,     0.,    3.   )
setter.set( 0 ,0, -1.99739    , 0.001,      -2.2,  -1.6 )
setter.set( 0 ,0, 1.59216     , 0.001,      1.,    2.2  )
setter.set( 1 ,0, 0.          , 0.0001,     0.,    0.01 )

process = cms.Process("TEST")

# process.source = cms.Source(
#     "PoolSource",
#     fileNames = cms.untracked.vstring(
#       "file:/home/demattia/MuScleFit/PatMuons/onia2MuMuPAT_Summer10-DESIGN_36_V8-X0MAX-v2.root"
#     )
# )
# process.maxEvents = cms.untracked.PSet(
#     input = cms.untracked.int32(-1)
# )

# Use this when running on a tree
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(0)
)

process.looper = cms.Looper(
    "MuScleFit",

    # Only used when reading events from a root tree
    MaxEventsFromRootTree = cms.int32(-1),
    # Specify a file if you want to read events from a root tree in a local file.
    # In this case the input source should be an empty source with 0 events.

    #InputRootTreeFileName = cms.string("/home/castello/7TeV/CMSSW_3_8_5_patch3/src/Tree/Tree_2010AB_INNtk_EOYgeom_BS.root"),
    #InputRootTreeFileName = cms.string("/home/castello/7TeV/CMSSW_3_8_5_patch3/src/Tree/Fall10/Tree_MCFall2010_INNtk_CRAFTRealistic_wGEN.root"),
    InputRootTreeFileName = cms.string("/home/castello/7TeV/CMSSW_3_8_5_patch3/src/Tree/Tree_2010AB_INNtk_ICHEPgeom_BS.root"),
    
    # Specify the file name where you want to save a root tree with the muon pairs.
    # Leave empty if no file should be written.
    OutputRootTreeFileName = cms.string(""), #Tree_38X_INNtk_resol42_test7.root

    # Choose the kind of muons you want to run on
    # -------------------------------------------
    MuonLabel = cms.InputTag("patMuons"),
    # Defines what type of muons to use:
    # -1 = onia guys selection
    # -2 = onia guys selection - only GG
    # -3 = onia guys selection - only GT
    # -4 = onia guys selection - only TT
    # Note that the above samples are independent and represent the composition of the inclusive sample
    # 1 = global muon
    # 2 = standalone muon
    # 3 = tracker muon
    # 4 = calo muon
    # 10 = innerTrack of global muon
    MuonType = cms.int32(-1),

    # This line allows to switch to PAT muons. Default is false.
    # Note that the onia selection works only with onia patTuples.
    PATmuons = cms.untracked.bool(True),

    # ---------------- #
    # Select resonance #
    # ---------------- #
    # The resonances are to be specified in this order:
    # Z0, Y(3S), Y(2S), Y(1S), Psi(2S), J/Psi
    # -------------------------------------------------
    resfind = cms.vint32(1, 0, 0, 0, 0, 0),

    # Likelihood settings
    # -------------------
    maxLoopNumber = cms.untracked.int32(2),
    # Select which fits to do in which loop (0 = do not, 1 = do)
    doResolFit =        cms.vint32(0,1),
    doScaleFit =        cms.vint32(1,0),
    doBackgroundFit =   cms.vint32(0,0),
    doCrossSectionFit = cms.vint32(0,0),

    # Use the probability file or not. If not it will perform a simpler selection taking the muon pair with
    # invariant mass closer to the pdf value and will crash if some fit is attempted.
    UseProbsFile = cms.untracked.bool(True),

    # False = use also MC information
    speedup = cms.bool(True),
    # Set this to false if you do not want to use simTracks.
    # (Note that this is skipped anyway if speedup == True).
    compareToSimTracks = cms.bool(False),

    # Output settings
    # ---------------
    OutputFileName = cms.untracked.string('MuScleFit_4Nov_INNtk_resol42_comb.root'),

    # BiasType=0 means no bias to muon momenta
    # ----------------------------------------
    BiasType = cms.int32(0),
    parBias = cms.vdouble(),

    # SmearType=0 means no smearing applied to muon momenta
    # -----------------------------------------------------
    SmearType = cms.int32(0),
    parSmear = cms.vdouble(),

### taken from J/Psi #########################
    ResolFitType = cms.int32(42),
    parResolFix = setter.parResolFix,
    parResolOrder = setter.parResolOrder,
    parResol = setter.parResol,
    parResolStep = setter.parResolStep,
    parResolMin = setter.parResolMin,
    parResolMax = setter.parResolMax,


    # -------------------- #
    # Scale fit parameters #
    # -------------------- #

    # -----------------------------------------------------------------------------------
    ScaleFitType = cms.int32(29),
    parScaleOrder = cms.vint32( 0, 0, 0, 0, 0),
    parScaleFix =   cms.vint32(1, 1, 1, 1, 1),
    parScale = cms.vdouble(-7.3019e-05, 0., 0.00147514, 0.000114635, 0.246663),
    
    # Scale fit type=11: Linear in pt, sinusoidal in phi with muon sign -->GOOD results in phi
    ##  modified for mu+/mu -
    # -----------------------------------------------------------------------------------
    ##    ScaleFitType = cms.int32(11),
    ##     parScaleOrder = cms.vint32(0, 0, 0, 0, 0, 0, 0, 0),
    ##     parScaleFix =   cms.vint32(0, 0, 0, 0, 0, 0, 0, 0),
    ##     parScale = cms.vdouble(1., 0., 0., 1., 0., 0., 1., 0.),


    # ---------------------------- #
    # Cross section fit parameters #
    # ---------------------------- #
    # Note that the cross section fit works differently than the others, it
    # fits ratios of parameters. Fix and Order should not be used as is, they
    # are there mainly for compatibility.
    parCrossSectionOrder = cms.vint32(0, 0, 0, 0, 0, 0),
    parCrossSectionFix =   cms.vint32(0, 0, 0, 0, 0, 0),
    parCrossSection = cms.vdouble(1.233, 2.07, 6.33, 13.9, 2.169, 127.2),

    # ------------------------- #
    # Background fit parameters #
    # ------------------------- #

    # Window factors for: Z, Upsilons and (J/Psi,Psi2S) regions
    LeftWindowBorder = cms.vdouble(70., 8., 1.391495),
    RightWindowBorder = cms.vdouble(110., 12., 5.391495),

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


    # ----------------------- #

    # Set Minuit fit strategy
    FitStrategy = cms.int32(2),

    # Fit accuracy and debug parameters
    StartWithSimplex = cms.bool(True),
    ComputeMinosErrors = cms.bool(False),
    MinimumShapePlots = cms.bool(True),

    # Set the cuts on muons to be used in the fit

    ##    MaxMuonPt = cms.untracked.double(50.),
    #MinMuonPt = cms.untracked.double(50.),
 ##    MinMuonEtaFirstRange = cms.untracked.double(-0.8),
##     MaxMuonEtaFirstRange = cms.untracked.double(0.8),
##     MinMuonEtaSecondRange = cms.untracked.double(-0.8),
##     MaxMuonEtaSecondRange = cms.untracked.double(0.8),
    
    # ProbabilitiesFileInPath = cms.untracked.string("MuonAnalysis/MomentumScaleCalibration/test/Probs_merge.root"),
    ProbabilitiesFile = cms.untracked.string("/home/castello/7TeV/CMSSW_3_8_5_patch3/src/MuonAnalysis/MomentumScaleCalibration/test/Probs_merge.root"),

    # The following parameters can be used to filter events
    TriggerResultsLabel = cms.untracked.string("TriggerResults"),
    TriggerResultsProcess = cms.untracked.string("HLT"),
    # TriggerPath: "" = No trigger requirements, "All" = No specific path
    #TriggerPath = cms.untracked.string("HLT_L1MuOpen"),
    TriggerPath = cms.untracked.string("All"),
    # Negate the result of the trigger
    NegateTrigger = cms.untracked.bool(False),

    debug = cms.untracked.int32(0),
)
