###### Universal configuration template for tracker alignment
#
#  Usage:
#
#    Make a copy of this file and insert Startgeometry, Alignables and
#    Pedesettings directly into it.
#
#    Specify the path to this config-Template in the alignment_setup.ini
#
#    The scripts mps_alisetup.py and mps_setup.py set the Variables at the top (setup*).
#
#        Collection specifies the type of Tracks. Currently these are supported:
#          - ALCARECOTkAlMinBias       -> Minimum Bias
#          - ALCARECOTkAlCosmicsCTF0T  -> Cosmics, either at 0T or 3.8T
#          - ALCARECOTkAlMuonIsolated  -> Isolated Muon
#          - ALCARECOTkAlZMuMu         -> Z decay to two Muons
#          - generalTracks             -> general tracks treated like Minimum Bias
#          - ALCARECOTkAlCosmicsInCollisions -> Cosmics taken during collisions
#
#        Globaltag specifies the detector conditions.
#        Parts of the Globaltag are overwritten in Startgeometry.
#
#        monitorFile and binaryFile are automatically set by mps_setup.
#        e.g. millePedeMonitor004.root and milleBinary004.dat
#
#        AlgoMode specifies mode of AlignmentProducer.algoConfig -> mille or pede
#        mille is default. Pede mode is automatically set when merge config is created by MPS
#
#        CosmicsDecoMode and CosmicsZeroTesla are only relevant if collection
#        is ALCARECOTkAlCosmicsCTF0T
#
#        If primaryWidth is bigger than 0.0 it overwrites
#        process.AlignmentProducer.algoConfig.TrajectoryFactory.ParticleProperties.PrimaryWidth = ...
#        if primaryWidth<=0.0 it has no effect at all.


import FWCore.ParameterSet.Config as cms
process = cms.Process("Alignment")

################################################################################
# Variables edited by mps_alisetup.py. Used in functions below.
# You can change them manually as well.
# ------------------------------------------------------------------------------
setupGlobaltag        = "placeholder_globaltag"
setupCollection       = "placeholder_collection"
setupCosmicsDecoMode  = False
setupCosmicsZeroTesla = False
setupPrimaryWidth     = -1.0
setupJson             = "placeholder_json"
setupRunStartGeometry = -1

################################################################################
# Variables edited by MPS (mps_setup and mps_merge). Be careful.
# ------------------------------------------------------------------------------
# Default is "mille". Gets changed to "pede" by mps_merge.
setupAlgoMode         = "mille"

# MPS looks specifically for the string "ISN" so don't change this.
setupMonitorFile      = "millePedeMonitorISN.root"
setupBinaryFile       = "milleBinaryISN.dat"

# Input files. Edited by mps_splice.py
readFiles = cms.untracked.vstring()
################################################################################


################################################################################
# General setup
# ------------------------------------------------------------------------------
import Alignment.MillePedeAlignmentAlgorithm.alignmentsetup.GeneralSetup as generalSetup
generalSetup.setup(process, setupGlobaltag, setupCosmicsZeroTesla)


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

##########################
## insert Startgeometry ##
##########################

# You can use tagwriter.setCondition() to overwrite conditions in globaltag
#
# Example:
# tagwriter.setCondition(process,
#       connect = "frontier://FrontierProd/CMS_CONDITIONS",
#       record = "TrackerAlignmentErrorExtendedRcd",
#       tag = "TrackerIdealGeometryErrorsExtended210_mc")


#######################
## insert Alignables ##
#######################

# # to run a high-level alignment on real data (including TOB centering; use
# # pixel-barrel centering for MC) of the whole tracker you can use the
# # following configuration:
#
# process.AlignmentProducer.ParameterBuilder.parameterTypes = [
#     "SelectorRigid,RigidBody",
#     ]
# # Define the high-level structure alignables
# process.AlignmentProducer.ParameterBuilder.SelectorRigid = cms.PSet(
#     alignParams = cms.vstring(
#         "TrackerTPBHalfBarrel,111111",
#         "TrackerTPEHalfCylinder,111111",
#         "TrackerTIBHalfBarrel,111111",
#         "TrackerTOBHalfBarrel,rrrrrr",
#         "TrackerTIDEndcap,111111",
#         "TrackerTECEndcap,111111"
#         )
#     )


#########################
## insert Pedesettings ##
#########################

# # reasonable pede settings are already defined in
# # 'confAliProducer.setConfiguration' above
# # if you want obtain alignment errors, use "inversion 3 0.8" as
# # process.AlignmentProducer.algoConfig.pedeSteerer.method and set
# # process.AlignmentProducer.saveApeToDB = True
# # a list of possible options is documented here:
# # http://www.desy.de/~kleinwrt/MP2/doc/html/option_page.html#sec-cmd
# # you can change pede settings as follows:
#
# import Alignment.MillePedeAlignmentAlgorithm.alignmentsetup.helper as helper
# helper.set_pede_option(process, "entries 50 10 2")


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
