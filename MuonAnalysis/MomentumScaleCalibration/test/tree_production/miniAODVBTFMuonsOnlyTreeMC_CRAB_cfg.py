import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing


process = cms.Process("TREE")
process.options = cms.untracked.PSet(wantSummary=cms.untracked.bool(True))
process.options = cms.untracked.PSet(SkipEvent = cms.untracked.vstring('ProductNotFound'))

### command-line options
options = VarParsing.VarParsing()

### eta ranges steerable
options.register('etaMax1',
                 2.4,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.float,
                 "eta max (muon1)")

options.register('etaMin1',
                 -2.4,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.float,
                 "eta min (muon1)")

options.register('etaMax2',
                 2.4,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.float,
                 "eta max (muon2)")

options.register('etaMin2',
                 -2.4,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.float,
                 "eta min (muon2)")


# next line is not working with CRAB
#options.parseArguments()
### end of options


# Messages
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 500

from CondCore.DBCommon.CondDBSetup_cfi import *
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

############  DATABASE conditions  ###########################
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')   

########### FILES  ####################################
process.source = cms.Source(
    "PoolSource",
#    fileNames = cms.untracked.vstring('file:test.root')
    fileNames = cms.untracked.vstring(
'/store/mc/Phys14DR/DYToMuMu_M-50_Tune4C_13TeV-pythia8/MINIAODSIM/PU40bx25_tsg_castor_PHYS14_25_V1-v2/00000/622CAFBA-BD9A-E411-BE11-002481E14FFC.root'    
)
    )
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

############ Zmumu GoldenSelection sequence ###############################################
# http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/CMSSW/ElectroWeakAnalysis/ZMuMu/python/ZMuMuGolden_cfi.py?revision=1.7&view=markup&sortby=date

###########################################################################################
# RUN2 muon selection
###########################################################################################
process.RunTwoMuons = cms.EDFilter("PATMuonSelector",
   src = cms.InputTag("slimmedMuons"),                                 
   cut = cms.string(
    'pt > 20 & abs(eta)<2.4' + 
    '&& (isPFMuon && (isGlobalMuon || isTrackerMuon) )'+
    '&& (pfIsolationR04().sumChargedHadronPt+max(0.,pfIsolationR04().sumNeutralHadronEt+pfIsolationR04().sumPhotonEt-0.50*pfIsolationR04().sumPUPt))/pt < 0.20'
   )
 )

# for isolation (tight WP)
# https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideMuonId#Muon_Isolation

#### FILL TREE  #######################

### helper class definition ###
#####################
class SetParameters:
#####################
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
### end of the  definition ###

        
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


TREEINPUTNAME=""
TREEOUTPUTNAME="zmumuTree.root"                               
process.looper = cms.Looper(
    "MuScleFit",
    # Only used when reading events from a root tree
    MaxEventsFromRootTree = cms.int32(-1),

    # Specify a file if you want to read events from a root tree in a local file.
    # In this case the input source should be an empty source with 0 events.
    
    InputRootTreeFileName = cms.string(TREEINPUTNAME),
    
    # Specify the file name where you want to save a root tree with the muon pairs.
    # Leave empty if no file should be written.
    
    OutputRootTreeFileName = cms.string(TREEOUTPUTNAME),
    

    # Choose the kind of muons you want to run on
    # -------------------------------------------
    MuonLabel = cms.InputTag("RunTwoMuons"),
    MuonType = cms.int32(11),

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
    maxLoopNumber = cms.untracked.int32(1),
    # Select which fits to do in which loop (0 = do not, 1 = do)
    doResolFit =        cms.vint32(0),
    doScaleFit =        cms.vint32(0),
    doBackgroundFit =   cms.vint32(0),
    doCrossSectionFit = cms.vint32(0),

    # Use the probability file or not. If not it will perform a simpler selection taking the muon pair with
    # invariant mass closer to the pdf value and will crash if some fit is attempted.
    UseProbsFile = cms.untracked.bool(False),

    # False = use also MC information
    speedup = cms.bool(False),
    GenParticlesName = cms.untracked.string("prunedGenParticles"),

    # Set this to false if you do not want to use simTracks.
    # (Note that this is skipped anyway if speedup == True).
    compareToSimTracks = cms.bool(False),

    # Output settings
    # ---------------
    OutputFileName = cms.untracked.string("zmumuHisto.root"),

    # BiasType=0 means no bias to muon momenta
    # ----------------------------------------
    BiasType = cms.int32(0),
    parBias = cms.vdouble(),

    # SmearType=0 means no smearing applied to muon momenta
    # -----------------------------------------------------
    SmearType = cms.int32(0),
    parSmear = cms.vdouble(),

    ### taken from J/Psi #########################
    ResolFitType = cms.int32(0),
    parResolFix   = setter.parResolFix,
    parResolOrder = setter.parResolOrder,
    parResol      = setter.parResol,
    parResolStep  = setter.parResolStep,
    parResolMin   = setter.parResolMin,
    parResolMax   = setter.parResolMax,


    # -------------------- #
    # Scale fit parameters #
    # -------------------- #

    # -----------------------------------------------------------------------------------
    ScaleFitType  = cms.int32(0),
    parScaleOrder = cms.vint32( 0, 0, 0, 0, 0),
    parScaleFix   = cms.vint32(1, 1, 1, 1, 1),
    parScale      = cms.vdouble(-7.3019e-05, 0., 0.00147514, 0.000114635, 0.246663),
    
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
    parCrossSectionFix   = cms.vint32(0, 0, 0, 0, 0, 0),
    parCrossSection      = cms.vdouble(1.233, 2.07, 6.33, 13.9, 2.169, 127.2),

    # ------------------------- #
    # Background fit parameters #
    # ------------------------- #

    # Window factors for: Z, Upsilons and (J/Psi,Psi2S) regions
    LeftWindowBorder  = cms.vdouble(70., 8., 1.391495),
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
    StartWithSimplex   = cms.bool(True),
    ComputeMinosErrors = cms.bool(False),
    MinimumShapePlots  = cms.bool(True),


    ########## TO BE ENABLED ################################
    # Set the cuts on muons to be used in the fit
    MinMuonPt = cms.untracked.double(0.),
    MaxMuonPt = cms.untracked.double(1000.),
    MinMuonEtaFirstRange = cms.untracked.double(options.etaMin1),
    MaxMuonEtaFirstRange = cms.untracked.double(options.etaMax1),
    MinMuonEtaSecondRange = cms.untracked.double(options.etaMin2),
    MaxMuonEtaSecondRange = cms.untracked.double(options.etaMax2),
    
    #ProbabilitiesFileInPath = cms.untracked.string("MuonAnalysis/MomentumScaleCalibration/test/Probs_merge.root"),
    #ProbabilitiesFile = cms.untracked.string("Probs_merge.root"),

    # Pile-Up related info 
    PileUpSummaryInfo = cms.untracked.InputTag("addPileupInfo"),
    PrimaryVertexCollection = cms.untracked.InputTag("offlineSlimmedPrimaryVertices"),

    # The following parameters can be used to filter events
    TriggerResultsLabel = cms.untracked.string("TriggerResults"),
    TriggerResultsProcess = cms.untracked.string("HLT"),
    TriggerPath = cms.untracked.vstring(""),
    # Negate the result of the trigger
    NegateTrigger = cms.untracked.bool(False),
    debug = cms.untracked.int32(0)
)


process.p = cms.Path(
    process.RunTwoMuons
    )

