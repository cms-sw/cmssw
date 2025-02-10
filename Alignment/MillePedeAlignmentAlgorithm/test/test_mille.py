import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("Alignment")

options = VarParsing.VarParsing()
options.register ('algoMode',
                  "mille", # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "algo mode")

options.register ('useLapack',
                  False, # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.bool,
                  "use lapack?")
options.parseArguments()

################################################################################
# Variables edited by mps_alisetup.py. Used in functions below.
# You can change them manually as well.
# ------------------------------------------------------------------------------
setupGlobaltag = "140X_dataRun3_ForTkAlReReco_v1"
setupCollection = "ALCARECOTkAlZMuMu"
setupCosmicsDecoMode  = False
setupCosmicsZeroTesla = False
setupPrimaryWidth     = -1.0
setupRecoGeometry     = "" # empty string defaults to DB
setupJson = ""
setupRunStartGeometry = 362350

################################################################################
# Variables edited by MPS (mps_setup and mps_merge). Be careful.
# ------------------------------------------------------------------------------
# Default is "mille". Gets changed to "pede" by mps_merge.
setupAlgoMode         = options.algoMode

# MPS looks specifically for the string "101" so don't change this.
setupMonitorFile      = "millePedeMonitor101.root"
setupBinaryFile       = "milleBinary101.dat"

# Input files. Edited by mps_splice.py
readFiles = cms.untracked.vstring()
readFiles.extend([
    '/store/data/Run2022G/Muon/ALCARECO/TkAlZMuMu-PromptReco-v1/000/362/362/00000/d6641b44-f4e4-4054-b5b0-f038e567c61e.root',
    '/store/data/Run2022G/Muon/ALCARECO/TkAlZMuMu-PromptReco-v1/000/362/433/00000/1f93221e-23ce-4731-906a-48c9fe405515.root',
    '/store/data/Run2022G/Muon/ALCARECO/TkAlZMuMu-PromptReco-v1/000/362/435/00000/df6e27d1-5367-4192-83ed-2be9303d7837.root',
    '/store/data/Run2022G/Muon/ALCARECO/TkAlZMuMu-PromptReco-v1/000/362/437/00000/7ce5bac8-0b29-40f3-a63b-fd0813d5678d.root',
    '/store/data/Run2022G/Muon/ALCARECO/TkAlZMuMu-PromptReco-v1/000/362/437/00000/ea6b065d-1912-491e-9cce-732eaf6fa038.root'])
################################################################################

################################################################################
# General setup
# ------------------------------------------------------------------------------
import Alignment.MillePedeAlignmentAlgorithm.alignmentsetup.GeneralSetup as generalSetup
generalSetup.setup(process, setupGlobaltag, setupCosmicsZeroTesla, setupRecoGeometry)

################################################################################
# setup alignment producer
# ------------------------------------------------------------------------------
import Alignment.MillePedeAlignmentAlgorithm.alignmentsetup.ConfigureAlignmentProducer as confAliProducer

confAliProducer.setConfiguration(process,
    collection   = setupCollection,
    mode         = setupAlgoMode,
    monitorFile  = setupMonitorFile,
    binaryFile   = setupBinaryFile,
    primaryWidth = setupPrimaryWidth,
    cosmicsZeroTesla = setupCosmicsZeroTesla)

################################################################################
# Overwrite some conditions in global tag
# ------------------------------------------------------------------------------
import Alignment.MillePedeAlignmentAlgorithm.alignmentsetup.SetCondition as tagwriter

#######################
## insert Alignables ##
#######################

# # to run a high-level alignment on real data (including TOB centering; use
# # pixel-barrel centering for MC) of the whole tracker you can use the
# # following configuration:
#
process.AlignmentProducer.ParameterBuilder.parameterTypes = [
     "SelectorRigid,RigidBody",
     #"SelectorBowed,BowedSurface", 
     #"SelectorTwoBowed,TwoBowedSurfaces",
     ]

# # Define the high-level structure alignables
process.AlignmentProducer.ParameterBuilder.SelectorRigid = cms.PSet(
     alignParams = cms.vstring(
         "TrackerP1PXBLadder,111111",
         "TrackerP1PXECPanel,111111",
         "TrackerTIBHalfBarrel,111111",
         "TrackerTOBHalfBarrel,rrrrrr",
         "TrackerTIDEndcap,111111",
         "TrackerTECEndcap,111111",
     )
)

process.AlignmentProducer.RunRangeSelection = [
    cms.PSet(
        RunRanges = cms.vstring(
            "362350",
            "362440",
            "362446",
            "362617",
            "362632",
            "362640",
            "362641",
            "362645",
            "362663",
            "362670",
            "362679",
            "362683",
            "362697",
            "362711",
            "362744"
        ),
        selector = cms.vstring(
            "TrackerP1PXBLadder,111111",
            "TrackerP1PXECPanel,111111"
        )
    ),
    
    cms.PSet(
        RunRanges = cms.vstring(
            "362350",
            "362520"
        ),
        selector = cms.vstring(
            "TrackerTIBHalfBarrel,111111",
            "TrackerTIDEndcap,111111",
            "TrackerTECEndcap,111111"
        )
    )
] # end of process.AlignmentProducer.RunRangeSelection

#########################
## insert Pedesettings ##
#########################

# # reasonable pede settings are already defined in
# # 'confAliProducer.setConfiguration' above
# #

if(options.algoMode == "pede"):
    if(options.useLapack):
        # LAPACK
        print("I am going to run fullLAPACK 3 0.8")
        process.AlignmentProducer.algoConfig.pedeSteerer.method = "fullLAPACK 3 0.8"
        process.AlignmentProducer.algoConfig.pedeSteerer.pedeCommand = "export OMP_STACKSIZE=20M; MKL_THREADING_LAYER=GNU; export OMP_NUM_THREADS=10; export MKL_NUM_THREADS=10; \
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cvmfs/projects.cern.ch/intelsw/oneAPI/linux/x86_64/2022/mkl/2022.1.0/lib/intel64;"
    else:
        # MINRES
        print("I am going to run sparseMINRES 6 0.8")
        process.AlignmentProducer.algoConfig.pedeSteerer.method = "sparseMINRES 6 0.8"
        process.AlignmentProducer.algoConfig.pedeSteerer.pedeCommand = "export OMP_STACKSIZE=20M; pede"
else:
    pass
        
# # if you want to obtain alignment errors, use the following setting:
# #
# # a list of possible options is documented here:
# # http://www.desy.de/~kleinwrt/MP2/doc/html/option_page.html#sec-cmd
# #
# # if you need to request a larger stack size for individual threads when
# # running pede, you can do this with this setting:
# #
# # you can change or drop pede options as follows:
#
import Alignment.MillePedeAlignmentAlgorithm.alignmentsetup.helper as helper
helper.set_pede_option(process, "threads 10")
helper.set_pede_option(process, "entries 100 10 2")
helper.set_pede_option(process, "skipemptycons")
helper.set_pede_option(process, "countrecords")

#################
## add filters ##
#################

# # please add any EDFilter here that should run before processing the event,
# # e.g. add the following lines to ensure that only 3.8T events are selected
#
# import Alignment.MillePedeAlignmentAlgorithm.alignmentsetup.helper as helper
# process.load("Alignment.CommonAlignment.magneticFieldFilter_cfi")
# process.magneticFieldFilter.magneticField = 38 # in units of kGauss (=0.1T)
# helper.add_filter(process, process.magneticFieldFilter)

################################################################################
# Mille-procedure
# ------------------------------------------------------------------------------
if setupAlgoMode == "mille":
    import Alignment.MillePedeAlignmentAlgorithm.alignmentsetup.MilleSetup as mille
    mille.setup(process,
                input_files        = readFiles,
                collection         = setupCollection,
                json_file          = setupJson,
                cosmics_zero_tesla = setupCosmicsZeroTesla,
                cosmics_deco_mode  = setupCosmicsDecoMode)

################################################################################
# Pede-procedure
# ------------------------------------------------------------------------------
else:
    # placeholers get replaced by mps_merge.py, which is called in mps_setup.pl
    merge_binary_files = ['placeholder_binaryList']
    merge_tree_files   = ['placeholder_treeList']

    import Alignment.MillePedeAlignmentAlgorithm.alignmentsetup.PedeSetup as pede
    pede.setup(process,
               binary_files = merge_binary_files,
               tree_files = merge_tree_files,
               run_start_geometry = setupRunStartGeometry)
