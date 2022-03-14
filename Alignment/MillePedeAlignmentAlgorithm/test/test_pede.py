import FWCore.ParameterSet.Config as cms
process = cms.Process("Alignment")

process.Tracer = cms.Service("Tracer")

setupGlobaltag = "121X_mcRun3_2021_realistic_forpp900GeV_v6"
setupCollection = "ALCARECOTkAlCosmicsCosmicTF0T"
setupCosmicsDecoMode = True
setupCosmicsZeroTesla = False
setupPrimaryWidth     = -1.0
setupJson             = "placeholder_json"
setupRunStartGeometry = 1
setupAlgoMode = "pede"
setupMonitorFile      = "millePedeMonitorISN.root"
setupBinaryFile       = "milleBinaryISN.dat"
readFiles = cms.untracked.vstring()

import Alignment.MillePedeAlignmentAlgorithm.alignmentsetup.GeneralSetup as generalSetup
generalSetup.setup(process, setupGlobaltag, setupCosmicsZeroTesla)


################################################################################
# setup alignment producer
################################################################################
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
################################################################################
import Alignment.MillePedeAlignmentAlgorithm.alignmentsetup.SetCondition as tagwriter

################################################################################
# insert Startgeometry 
################################################################################
# You can use tagwriter.setCondition() to overwrite conditions in globaltag

################################################################################
# insert Alignables 
################################################################################
process.AlignmentProducer.ParameterBuilder.parameterTypes = ["SelectorRigid,RigidBody"]

################################################################################
# Define the high-level structure alignables
################################################################################
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

################################################################################
# insert Pedesettings 
################################################################################
import Alignment.MillePedeAlignmentAlgorithm.alignmentsetup.helper as helper
helper.set_pede_option(process, "skipemptycons")

################################################################################
# Mille-procedure
################################################################################
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
################################################################################
else:
    merge_binary_files = [
                'milleBinary001.dat',
                'milleBinary002.dat',
                'milleBinary003.dat']
    merge_tree_files   = [
                'treeFile001.root',
                'treeFile002.root',
                'treeFile003.root']

    import Alignment.MillePedeAlignmentAlgorithm.alignmentsetup.PedeSetup as pede
    pede.setup(process,
               binary_files = merge_binary_files,
               tree_files = merge_tree_files,
               run_start_geometry = setupRunStartGeometry)
