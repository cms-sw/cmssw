ZMuMuValidationTemplate="""

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
    resfind = cms.vint32(
      int(".oO[resonance]Oo." == "Z"),
      int(".oO[resonance]Oo." == "Y3S"),
      int(".oO[resonance]Oo." == "Y2S"),
      int(".oO[resonance]Oo." == "Y1S"),
      int(".oO[resonance]Oo." == "Psi2S"),
      int(".oO[resonance]Oo." == "JPsi")
    ),

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
    # Use compressions settings of TFile
    # see https://root.cern.ch/root/html534/TFile.html#TFile:SetCompressionSettings
    # settings = 100 * algorithm + level
    # level is from 1 (small) to 9 (large compression)
    # algo: 1 (ZLIB), 2 (LMZA)
    # see more about compression & performance: https://root.cern.ch/root/html534/guides/users-guide/InputOutput.html#compression-and-performance

    OutputFileName = cms.untracked.string("zmumuHisto.root"),
    compressionSettings = cms.untracked.int32(compressionSettings), 

    # BiasType=0 means no bias to muon momenta
    # ----------------------------------------
    BiasType = cms.int32(0),
    parBias = cms.vdouble(),

    # SmearType=0 means no smearing applied to muon momenta
    # -----------------------------------------------------
    SmearType = cms.int32(0),
    parSmear = cms.vdouble(),

    ### taken from J/Psi #########################
#    ResolFitType = cms.int32(14),
#    parResol = cms.vdouble(0.007,0.015, -0.00077, 0.0063, 0.0018, 0.0164),
#    parResolFix = cms.vint32(0, 0, 0,0, 0,0),
#    parResolOrder = cms.vint32(0, 0, 0, 0, 0, 0),
    ResolFitType = cms.int32(0),
    parResol = cms.vdouble(0),
    parResolFix = cms.vint32(0),
    parResolOrder = cms.vint32(0),


    # -------------------- #
    # Scale fit parameters #
    # -------------------- #

    # -----------------------------------------------------------------------------------
#    ScaleFitType = cms.int32(18),
#    parScaleOrder = cms.vint32(0, 0, 0, 0),
#    parScaleFix =   cms.vint32(0, 0, 0, 0),
#    parScale = cms.vdouble(1, 1, 1, 1),
    ScaleFitType = cms.int32(0),
    parScaleOrder = cms.vint32(0),
    parScaleFix =   cms.vint32(0),
    parScale = cms.vdouble(0),



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
    MinMuonPt = cms.untracked.double(.oO[minpt]Oo.),
    MaxMuonPt = cms.untracked.double(.oO[maxpt]Oo.),
    MinMuonEtaFirstRange = cms.untracked.double(.oO[etaminneg]Oo.),
    MaxMuonEtaFirstRange = cms.untracked.double(.oO[etamaxneg]Oo.),
    MinMuonEtaSecondRange = cms.untracked.double(.oO[etaminpos]Oo.),
    MaxMuonEtaSecondRange = cms.untracked.double(.oO[etamaxpos]Oo.),
    PileUpSummaryInfo = cms.untracked.InputTag("addPileupInfo"),
    PrimaryVertexCollection = cms.untracked.InputTag("offlinePrimaryVertices"),

    # The following parameters can be used to filter events
    TriggerResultsLabel = cms.untracked.string("TriggerResults"),
    TriggerResultsProcess = cms.untracked.string("HLT"),
    TriggerPath = cms.untracked.vstring(""),
    # Negate the result of the trigger
    NegateTrigger = cms.untracked.bool(False),
    debug = cms.untracked.int32(0),
)

"""


####################################################################
####################################################################
LoadMuonModules = """
process.load("RecoMuon.DetLayers.muonDetLayerGeometry_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("RecoMuon.TrackingTools.MuonServiceProxy_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
"""


####################################################################
####################################################################
ZMuMuPath = """
process.p = cms.Path(
    process.offlineBeamSpot*process.TrackRefitter
    )
"""


####################################################################
####################################################################
zMuMuScriptTemplate="""#!/bin/bash
source /afs/cern.ch/cms/caf/setup.sh
export X509_USER_PROXY=.oO[scriptsdir]Oo./.user_proxy

echo  -----------------------
echo  Job started at `date`
echo  -----------------------

cwd=`pwd`
cd .oO[CMSSW_BASE]Oo./src
export SCRAM_ARCH=.oO[SCRAM_ARCH]Oo.
eval `scram runtime -sh`
cd $cwd

mkdir -p .oO[datadir]Oo.
mkdir -p .oO[workingdir]Oo.
mkdir -p .oO[logdir]Oo.
rm -f .oO[logdir]Oo./*.stdout
rm -f .oO[logdir]Oo./*.stderr

if [[ $HOSTNAME = lxplus[0-9]*[.a-z0-9]* ]] # check for interactive mode
then
    mkdir -p .oO[workdir]Oo.
    rm -f .oO[workdir]Oo./*
    cd .oO[workdir]Oo.
else
    mkdir -p $cwd/TkAllInOneTool
    cd $cwd/TkAllInOneTool
fi


.oO[CommandLine]Oo.

ls -lh .

cp .oO[MuonAnalysis/MomentumScaleCalibration]Oo./test/Macros/RooFit/CompareBias.oO[resonance]Oo.Validation.cc .
cp .oO[MuonAnalysis/MomentumScaleCalibration]Oo./test/Macros/RooFit/Legend.h .
cp .oO[MuonAnalysis/MomentumScaleCalibration]Oo./test/Macros/RooFit/FitMassSlices.cc .
cp .oO[MuonAnalysis/MomentumScaleCalibration]Oo./test/Macros/RooFit/FitSlices.cc .
cp .oO[MuonAnalysis/MomentumScaleCalibration]Oo./test/Macros/RooFit/FitXslices.cc .
cp .oO[MuonAnalysis/MomentumScaleCalibration]Oo./test/Macros/RooFit/FitWithRooFit.cc .
cp .oO[MuonAnalysis/MomentumScaleCalibration]Oo./test/Macros/RooFit/FitMass1D.cc .

root -q -b -l "CompareBias.oO[resonance]Oo.Validation.cc+(.oO[rebinphi]Oo., .oO[rebinetadiff]Oo., .oO[rebineta]Oo., .oO[rebinpt]Oo.)"

cp  .oO[MuonAnalysis/MomentumScaleCalibration]Oo./test/Macros/RooFit/tdrstyle.C .
cp  .oO[MuonAnalysis/MomentumScaleCalibration]Oo./test/Macros/RooFit/MultiHistoOverlap_.oO[resonance]Oo..C .

if [[ .oO[zmumureference]Oo. == *store* ]]; then xrdcp -f .oO[zmumureference]Oo. BiasCheck_Reference.root; else ln -fs .oO[zmumureference]Oo. ./BiasCheck_Reference.root; fi
root -q -b -l MultiHistoOverlap_.oO[resonance]Oo..C

eos mkdir -p /store/group/alca_trackeralign/AlignmentValidation/.oO[eosdir]Oo./plots/
for RootOutputFile in $(ls *root )
do
    xrdcp -f ${RootOutputFile}  root://eoscms//eos/cms/store/group/alca_trackeralign/AlignmentValidation/.oO[eosdir]Oo./
    cp ${RootOutputFile}  .oO[workingdir]Oo.
done

mkdir -p .oO[plotsdir]Oo.
for PngOutputFile in $(ls *png ); do
    xrdcp -f ${PngOutputFile}  root://eoscms//eos/cms/store/group/alca_trackeralign/AlignmentValidation/.oO[eosdir]Oo./plots/
    cp ${PngOutputFile}  .oO[plotsdir]Oo.
done


echo  -----------------------
echo  Job ended at `date`
echo  -----------------------

"""

######################################################################
######################################################################

mergeZmumuPlotsExecution="""
#merge Z->mumu histograms

cp .oO[mergeZmumuPlotsScriptPath]Oo. .
root -l -x -b -q TkAlMergeZmumuPlots.C++

"""

######################################################################
######################################################################

mergeZmumuPlotsTemplate="""
#include "MuonAnalysis/MomentumScaleCalibration/test/Macros/RooFit/MultiHistoOverlapAll_.oO[resonance]Oo..C"
#include <sstream>
#include <vector>

template <typename T> string separatebycommas(vector<T> v){
  if (v.size()==0) return "";
  stringstream s;
  s << v[0];
  for (unsigned int i = 1; i < v.size(); i++) s << "," << v[i];
  return s.str();
}

void TkAlMergeZmumuPlots(){
  vector<string> filenames; vector<string> titles; vector<int> colors; vector<int> linestyles;

.oO[PlottingInstantiation]Oo.

  vector<int> linestyles_new, markerstyles_new;
  for (unsigned int j=0; j<linestyles.size(); j++){ linestyles_new.push_back(linestyles.at(j) % 100); markerstyles_new.push_back(linestyles.at(j) / 100); }

  TkAlStyle::legendheader = ".oO[legendheader]Oo.";
  TkAlStyle::set(.oO[publicationstatus]Oo., .oO[era]Oo., ".oO[customtitle]Oo.", ".oO[customrighttitle]Oo.");

  MultiHistoOverlapAll_.oO[resonance]Oo.(separatebycommas(filenames), separatebycommas(titles), separatebycommas(colors), separatebycommas(linestyles_new), separatebycommas(markerstyles_new), ".oO[datadir]Oo./.oO[PlotsDirName]Oo.", .oO[switchONfit]Oo., .oO[AutoSetRange]Oo., .oO[CustomMinY]Oo., .oO[CustomMaxY]Oo.);
}
"""
