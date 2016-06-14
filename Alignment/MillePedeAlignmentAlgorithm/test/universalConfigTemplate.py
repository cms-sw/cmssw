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

# # typical pede settings are listed below;
# # if you want obtain alignment errors, use "inversion 3 0.8" as
# # process.AlignmentProducer.algoConfig.pedeSteerer.method and set
# # process.AlignmentProducer.saveApeToDB = True
#
# process.AlignmentProducer.algoConfig.pedeSteerer.method = "sparseMINRES-QLP 3  0.8"
# process.AlignmentProducer.algoConfig.pedeSteerer.options = [
#     "entries 50 10 2",
#     "outlierdownweighting 3",
#     "dwfractioncut 0.1",
#     "compress",
#     "threads 10",
#     "matiter 1",
#     "printcounts 2",
#     "chisqcut  30.  6.",
#     "bandwidth 6 1",
#     "monitorresiduals",
# ]
# process.AlignmentProducer.algoConfig.minNumHits = 8


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
               tree_files = merge_tree_files)
