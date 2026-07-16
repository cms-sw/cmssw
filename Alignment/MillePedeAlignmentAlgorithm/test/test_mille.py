import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

import Configuration.Geometry.defaultPhase2ConditionsEra_cff as _settings
_PH2_GLOBAL_TAG, _PH2_ERA = _settings.get_era_and_conditions(_settings.DEFAULT_VERSION)
process = cms.Process("Alignment", _PH2_ERA)

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
setupGlobaltag = _PH2_GLOBAL_TAG
setupCollection = "ALCARECOTkAlZMuMu"
setupCosmicsDecoMode  = False
setupCosmicsZeroTesla = False
setupPrimaryWidth     = -1.0
setupRecoGeometry     = "ExtendedRun4Default" # empty string defaults to DB
setupJson = ""
setupRunStartGeometry = 1

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
readFiles.extend(['/store/relval/CMSSW_20_0_0_pre1/RelValZMM_14/ALCARECO/TkAlZMuMu-150X_mcRun4_realistic_v1_STD_RegeneratedGS_D121_noPU-v1/2590000/d42b4f5d-7e50-461a-924c-9e95ad81194b.root'])
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
# Configure the MessageLogger service to dump information to cout (instead of alignment.log)
# see https://github.com/cms-sw/cmssw/issues/47963 for more information
# ------------------------------------------------------------------------------
process.MessageLogger.destinations = cms.untracked.vstring('cout')
process.MessageLogger.statistics = cms.untracked.vstring('cout')
# Copy all parameters from the existing 'alignment' PSet to a new 'cout' PSet
if hasattr(process.MessageLogger, 'alignment'):
    alignment_pset = process.MessageLogger.alignment
    process.MessageLogger.cout = alignment_pset

    # Optionally delete the old 'alignment' PSet
    delattr(process.MessageLogger, 'alignment')

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
         "TrackerP2PXBLadder,111111",
         "TrackerP2PXECPanel,111111",
         "TrackerP2OTBHalfBarrel,rrrrrr",
         "TrackerP2OTECEndcap,111111",
     )
)

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
