import FWCore.ParameterSet.Config as cms
process = cms.Process("Alignment")

################################################################################
# Variables edited by mps_alisetup.py. Used in functions below.
# You can change them manually as well.
# ------------------------------------------------------------------------------
setupGlobaltag = "122X_dataRun3_Prompt_v3"
setupCollection = "ALCARECOTkAlCosmicsCosmicTF0T"
setupCosmicsDecoMode = True
setupCosmicsZeroTesla = False
setupPrimaryWidth     = -1.0
setupJson             = "placeholder_json"
setupRunStartGeometry = 348908

################################################################################
# Variables edited by MPS (mps_setup and mps_merge). Be careful.
# ------------------------------------------------------------------------------
# Default is "mille". Gets changed to "pede" by mps_merge.
setupAlgoMode = "pede"

# MPS looks specifically for the string "001" so don't change this.
setupMonitorFile      = "millePedeMonitor001.root"
setupBinaryFile       = "milleBinary001.dat"

# Input files. Edited by mps_splice.py
readFiles = cms.untracked.vstring()
readFiles.extend([
    '/store/data/Commissioning2022/Cosmics/ALCARECO/TkAlCosmics0T-PromptReco-v1/000/348/268/00000/863844bd-0350-4131-8ef0-bc2fc1c6cb85.root'])
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

#######################
## insert Alignables ##
#######################

# # to run a high-level alignment on real data (including TOB centering; use
# # pixel-barrel centering for MC) of the whole tracker you can use the
# # following configuration:
##
process.AlignmentProducer.ParameterBuilder.parameterTypes = [
    "SelectorRigid,RigidBody"
    ]
#
# # Define the high-level structure alignables
process.AlignmentProducer.ParameterBuilder.SelectorRigid = cms.PSet(
    alignParams = cms.vstring(
        "TrackerP1PXBHalfBarrel,111111",
        "TrackerP1PXECHalfCylinder,111111",
        "TrackerTIBHalfBarrel,111111",
        "TrackerTOBHalfBarrel,rrrrrr",
        "TrackerTIDEndcap,111111",
        "TrackerTECEndcap,111111",
        )
    )

#########################
## insert Pedesettings ##
#########################
import Alignment.MillePedeAlignmentAlgorithm.alignmentsetup.helper as helper
helper.set_pede_option(process, "entries 100 10 2")
# helper.set_pede_option(process, "compress", drop = True)

import Alignment.MillePedeAlignmentAlgorithm.alignmentsetup.helper as helper
helper.set_pede_option(process, "skipemptycons")

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
    merge_binary_files = [
        'milleBinary055.dat',
        'milleBinary056.dat',
        'milleBinary057.dat',
        'milleBinary058.dat',
        'milleBinary059.dat',
        'milleBinary076.dat',
        'milleBinary077.dat',
        'milleBinary078.dat',
        'milleBinary079.dat',
        'milleBinary096.dat',
        'milleBinary098.dat',
        'milleBinary099.dat',
        'milleBinary100.dat',
        'milleBinary101.dat',
        'milleBinary134.dat',
        'milleBinary135.dat',
        'milleBinary136.dat',
        'milleBinary137.dat',
        'milleBinary138.dat',
        'milleBinary139.dat',
        'milleBinary140.dat',
        'milleBinary141.dat',
        'milleBinary142.dat',
        'milleBinary143.dat',
        'milleBinary144.dat',
        'milleBinary145.dat',
        'milleBinary146.dat',
        'milleBinary147.dat',
        'milleBinary149.dat',
        'milleBinary150.dat',
        'milleBinary151.dat',
        'milleBinary152.dat',
        'milleBinary153.dat',
        'milleBinary154.dat',
        'milleBinary163.dat',
        'milleBinary164.dat',
        'milleBinary165.dat',
        'milleBinary166.dat',
        'milleBinary167.dat',
        'milleBinary168.dat',
        'milleBinary177.dat',
        'milleBinary180.dat',
        'milleBinary182.dat',
        'milleBinary183.dat',
        'milleBinary184.dat',
        'milleBinary185.dat',
        'milleBinary186.dat',
        'milleBinary187.dat']
    merge_tree_files = [
        'treeFile055.dat',
        'treeFile056.dat',
        'treeFile057.dat',
        'treeFile058.dat',
        'treeFile059.dat',
        'treeFile076.dat',
        'treeFile077.dat',
        'treeFile078.dat',
        'treeFile079.dat',
        'treeFile096.dat',
        'treeFile098.dat',
        'treeFile099.dat',
        'treeFile100.dat',
        'treeFile101.dat',
        'treeFile134.dat',
        'treeFile135.dat',
        'treeFile136.dat',
        'treeFile137.dat',
        'treeFile138.dat',
        'treeFile139.dat',
        'treeFile140.dat',
        'treeFile141.dat',
        'treeFile142.dat',
        'treeFile143.dat',
        'treeFile144.dat',
        'treeFile145.dat',
        'treeFile146.dat',
        'treeFile147.dat',
        'treeFile149.dat',
        'treeFile150.dat',
        'treeFile151.dat',
        'treeFile152.dat',
        'treeFile153.dat',
        'treeFile154.dat',
        'treeFile163.dat',
        'treeFile164.dat',
        'treeFile165.dat',
        'treeFile166.dat',
        'treeFile167.dat',
        'treeFile168.dat',
        'treeFile177.dat',
        'treeFile180.dat',
        'treeFile182.dat',
        'treeFile183.dat',
        'treeFile184.dat',
        'treeFile185.dat',
        'treeFile186.dat',
        'treeFile187.dat']

    import Alignment.MillePedeAlignmentAlgorithm.alignmentsetup.PedeSetup as pede
    pede.setup(process,
               binary_files = merge_binary_files,
               tree_files = merge_tree_files,
               run_start_geometry = setupRunStartGeometry)

import Alignment.MillePedeAlignmentAlgorithm.alignmentsetup.SetCondition as tagwriter

tagwriter.setCondition(process,
       connect = "sqlite_file:alignment_input.db",
       record = "TrackerAlignmentRcd",
       tag = "TrackerAlignment_PCL_byRun_v2_express_348155")

tagwriter.setCondition(process,
       connect = "sqlite_file:alignment_input.db",
       record = "TrackerSurfaceDeformationRcd",
       tag = "TrackerSurafceDeformations_v1_express_299685")

tagwriter.setCondition(process,
       connect = "sqlite_file:alignment_input.db",
       record = "TrackerAlignmentErrorExtendedRcd",
       tag = "TrackerAlignmentExtendedErr_2009_v2_express_IOVs_347303")
