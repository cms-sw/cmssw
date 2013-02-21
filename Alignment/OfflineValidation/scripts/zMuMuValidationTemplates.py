ZMuMuValidationTemplate="""
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("ONLYHISTOS")

#process.options = cms.untracked.PSet(SkipEvent = cms.untracked.vstring('ProductNotFound'))

### command-line options
options = VarParsing.VarParsing()

### eta ranges steerable
options.register('etaMax1',
                 .oO[etamax1]Oo.,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.float,
                 "eta max (muon1)")

options.register('etaMin1',
                 .oO[etamin1]Oo.,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.float,
                 "eta min (muon1)")

options.register('etaMax2',
                 .oO[etamax2]Oo.,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.float,
                 "eta max (muon2)")

options.register('etaMin2',
                 .oO[etamin2]Oo.,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.float,
                 "eta min (muon2)")


options.parseArguments()

### end of options


# Messages
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.destinations = ['cout', 'cerr']
process.MessageLogger.cerr.FwkReport.reportEvery = 5000


########### DATA FILES  ####################################
process.load("Alignment.OfflineValidation..oO[dataset]Oo._cff")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.CommonDetUnit.globalTrackingGeometry_cfi")
process.load("RecoMuon.DetLayers.muonDetLayerGeometry_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("RecoMuon.TrackingTools.MuonServiceProxy_cff")

########### standard includes ##############################
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.Geometry_cff")

########### DATABASE conditions ############################
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = ".oO[GlobalTag]Oo."

.oO[dbLoad]Oo.

.oO[condLoad]Oo.

.oO[APE]Oo.

########### TRACK REFITTER #################################
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
process.TrackRefitter.src = 'ALCARECOTkAlZMuMu'
process.TrackRefitter.TrajectoryInEvent = True
process.TrackRefitter.TTRHBuilder = "WithAngleAndTemplate"

###### MuSclFit SETTINGS  ##############################################


### MuScleFit specific configuration 
 
process.looper = cms.Looper(
    "MuScleFit",
    # Only used when reading events from a root tree
    MaxEventsFromRootTree = cms.int32(-1),

    # Specify a file if you want to read events from a root tree in a local file.
    # In this case the input source should be an empty source with 0 events.
    
    InputRootTreeFileName = cms.string(""),
    
    # Specify the file name where you want to save a root tree with the muon pairs.
    # Leave empty if no file should be written.
    
    OutputRootTreeFileName = cms.string(""),
    

    # Choose the kind of muons you want to run on
    # -------------------------------------------
    MuonLabel = cms.InputTag("TrackRefitter"),


    #MuonType = cms.int32(11),
    MuonType = cms.int32(5),

    # This line allows to switch to PAT muons. Default is false.
    # Note that the onia selection works only with onia patTuples.
    PATmuons = cms.untracked.bool(False),

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
    speedup = cms.bool(True),
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
    ResolFitType = cms.int32(14), 
    parResol = cms.vdouble(0.007,0.015, -0.00077, 0.0063, 0.0018, 0.0164),
    parResolFix = cms.vint32(0, 0, 0,0, 0,0),
    parResolOrder = cms.vint32(0, 0, 0, 0, 0, 0),


    # -------------------- #
    # Scale fit parameters #
    # -------------------- #

    # -----------------------------------------------------------------------------------
    ScaleFitType = cms.int32(18),
    parScaleOrder = cms.vint32(0, 0, 0, 0),
    parScaleFix =   cms.vint32(0, 0, 0, 0),
    parScale = cms.vdouble(1, 1, 1, 1),


    
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
    FitStrategy = cms.int32(1),


    # Fit accuracy and debug parameters
    StartWithSimplex = cms.bool(True),
    ComputeMinosErrors = cms.bool(False),
    MinimumShapePlots = cms.bool(True),

    ########## TO BE ENABLED ################################
    # Set the cuts on muons to be used in the fit
    MinMuonPt = cms.untracked.double(0.),
    MaxMuonPt = cms.untracked.double(1000.),
    MinMuonEtaFirstRange = cms.untracked.double(options.etaMin1),
    MaxMuonEtaFirstRange = cms.untracked.double(options.etaMax1),
    MinMuonEtaSecondRange = cms.untracked.double(options.etaMin2),
    MaxMuonEtaSecondRange = cms.untracked.double(options.etaMax2),
    
    # The following parameters can be used to filter events
    TriggerResultsLabel = cms.untracked.string("TriggerResults"),
    TriggerResultsProcess = cms.untracked.string("HLT"),
    TriggerPath = cms.untracked.vstring(""),
    # Negate the result of the trigger
    NegateTrigger = cms.untracked.bool(False),
    debug = cms.untracked.int32(0),
)

###### FINAL SEQUENCE ##############################################

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(.oO[nEvents]Oo.))

process.p = cms.Path(
    process.offlineBeamSpot*process.TrackRefitter
    )
    
"""


####################################################################
####################################################################
zMuMuScriptTemplate="""
#!/bin/bash 
source /afs/cern.ch/cms/caf/setup.sh

echo  -----------------------
echo  Job started at `date`
echo  -----------------------

cwd=`pwd`
cd .oO[CMSSW_BASE]Oo./src
# export SCRAM_ARCH=slc5_amd64_gcc462
export SCRAM_ARCH=.oO[SCRAM_ARCH]Oo.
eval `scram runtime -sh`
cd $cwd

rfmkdir -p .oO[datadir]Oo.

rfmkdir -p .oO[logdir]Oo.
rm -f .oO[logdir]Oo./*.stdout
rm -f .oO[logdir]Oo./*.stderr

## The lines related to the 'workdir' are commented out in order to avoid issues with removed files from the /tmp directory.
# rfmkdir -p .oO[workdir]Oo.
# rm -f .oO[workdir]Oo./*
# cd .oO[workdir]Oo.


.oO[CommandLine]Oo.

ls -lh . 

source /afs/cern.ch/sw/lcg/external/gcc/4.6.2/x86_64-slc5/setup.sh
source /afs/cern.ch/sw/lcg/app/releases/ROOT/5.32.00/x86_64-slc5-gcc46-opt/root/bin/thisroot.sh

# cd .oO[CMSSW_BASE]Oo./src/MuonAnalysis/MomentumScaleCalibration/test/Macros/RooFit
# ln -fs .oO[workdir]Oo./0_zmumuHisto.root .
cp .oO[CMSSW_BASE]Oo./src/MuonAnalysis/MomentumScaleCalibration/test/Macros/RooFit/CompareBiasZValidation.cc .
cp .oO[CMSSW_BASE]Oo./src/MuonAnalysis/MomentumScaleCalibration/test/Macros/RooFit/Legend.h .
cp .oO[CMSSW_BASE]Oo./src/MuonAnalysis/MomentumScaleCalibration/test/Macros/RooFit/FitMassSlices.cc .
cp .oO[CMSSW_BASE]Oo./src/MuonAnalysis/MomentumScaleCalibration/test/Macros/RooFit/FitSlices.cc .
cp .oO[CMSSW_BASE]Oo./src/MuonAnalysis/MomentumScaleCalibration/test/Macros/RooFit/FitXslices.cc .
cp .oO[CMSSW_BASE]Oo./src/MuonAnalysis/MomentumScaleCalibration/test/Macros/RooFit/FitWithRooFit.cc .

root -q -b "CompareBiasZValidation.cc+(\\\"\\\")"

 
# mv BiasCheck.root .oO[workdir]Oo. 

# cd .oO[workdir]Oo.
cp  .oO[CMSSW_BASE]Oo./src/MuonAnalysis/MomentumScaleCalibration/test/Macros/RooFit/tdrstyle.C .
cp  .oO[CMSSW_BASE]Oo./src/MuonAnalysis/MomentumScaleCalibration/test/Macros/RooFit/MultiHistoOverlap.C .
# ln -fs /afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN2/TMP_EM/ZMuMu/data/MC/BiasCheck_DYToMuMu_Summer11_TkAlZMuMu_IDEAL.root  ./BiasCheck_Reference.root
ln -fs .oO[zmumureference]Oo. ./BiasCheck_Reference.root
root -q -b MultiHistoOverlap.C

for RootOutputFile in $(ls *root ); do
    rfcp ${RootOutputFile}  .oO[datadir]Oo.
done

for PngOutputFile in $(ls *png ); do
    rfcp ${PngOutputFile}  .oO[datadir]Oo.
done


echo  -----------------------
echo  Job ended at `date`
echo  -----------------------    

"""
