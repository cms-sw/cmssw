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
#          - ALCARECOTkAlUpsilonMuMu   -> Upsilon decay to two Muons
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

# # You can use tagwriter.setCondition() to overwrite conditions in globaltag
# #
# # Examples (ideal phase-1 tracker-alignment conditions):
# tagwriter.setCondition(process,
#       connect = "frontier://FrontierProd/CMS_CONDITIONS",
#       record = "TrackerAlignmentRcd",
#       tag = "TrackerAlignment_Upgrade2017_design_v4")
# tagwriter.setCondition(process,
#       connect = "frontier://FrontierProd/CMS_CONDITIONS",
#       record = "TrackerSurfaceDeformationRcd",
#       tag = "TrackerSurfaceDeformations_zero")
# tagwriter.setCondition(process,
#       connect = "frontier://FrontierProd/CMS_CONDITIONS",
#       record = "TrackerAlignmentErrorExtendedRcd",
#       tag = "TrackerAlignmentErrorsExtended_Upgrade2017_design_v0")
# tagwriter.setCondition(process,
#       connect = "frontier://FrontierProd/CMS_CONDITIONS",
#       record = "SiPixelLorentzAngleRcd",
#       label = "fromAlignment",
#       tag = "SiPixelLorentzAngle_fromAlignment_phase1_mc_v1")


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
#
# # Define the high-level structure alignables
# process.AlignmentProducer.ParameterBuilder.SelectorRigid = cms.PSet(
#     alignParams = cms.vstring(
#         "TrackerP1PXBHalfBarrel,111111",
#         "TrackerP1PXECHalfCylinder,111111",
#         "TrackerTIBHalfBarrel,111111",
#         "TrackerTOBHalfBarrel,rrrrrr",
#         "TrackerTIDEndcap,111111",
#         "TrackerTECEndcap,111111",
#     )
# )


# # to run a module-level alignment on real data (including TOB centering; use
# # pixel-barrel centering for MC) of the whole tracker (including surface
# # deformations) you can use the following configuration (read comments on
# # multi-IOV alignment below):
#
# process.AlignmentProducer.ParameterBuilder.parameterTypes = [
#     "SelectorRigid,RigidBody",
#     "SelectorBowed,BowedSurface",
#     "SelectorTwoBowed,TwoBowedSurfaces",
# ]
#
# # Define the high-level structure alignables
# process.AlignmentProducer.ParameterBuilder.SelectorRigid = cms.PSet(
#     alignParams = cms.vstring(
#         "TrackerP1PXBHalfBarrel,111111",
#         "TrackerP1PXECHalfCylinder,111111",
#         "TrackerTIBHalfBarrel,111111",
#         "TrackerTOBHalfBarrel,rrrrrr",
#         "TrackerTIDEndcap,111111",
#         "TrackerTECEndcap,111111",
#     )
# )
#
# # Define the module-level alignables (for single modules)
# process.AlignmentProducer.ParameterBuilder.SelectorBowed = cms.PSet(
#     alignParams = cms.vstring(
#         "TrackerP1PXBModule,111111 111",
#         "TrackerP1PXECModule,111111 111",
#         "TrackerTIBModuleUnit,101111 111",
#         "TrackerTIDModuleUnit,111111 111",
#         "TrackerTECModuleUnit,111111 111,tecSingleSens",
#     ),
#     tecSingleSens = cms.PSet(tecDetId = cms.PSet(ringRanges = cms.vint32(1,4))),
# )
#
# process.AlignmentProducer.ParameterBuilder.SelectorTwoBowed = cms.PSet(
#     alignParams = cms.vstring(
#         "TrackerTOBModuleUnit,101111 111 101111 111",
#         "TrackerTECModuleUnit,111111 111 111111 111,tecDoubleSens",
#     ),
#     tecDoubleSens = cms.PSet(tecDetId = cms.PSet(ringRanges = cms.vint32(5,7))),
# )
#
# # IOV definition
# #  - defaults to single-IOV starting at "1", if omitted
# #  - alignables have to match high-level structures above
# #    -> except for 'rrrrrr' alignables
# process.AlignmentProducer.RunRangeSelection = [
#     cms.PSet(
#         RunRanges = cms.vstring(
#             "290550",
#             "300000",
#         ),
#         selector = cms.vstring(
#             "TrackerP1PXBHalfBarrel,111111",
#             "TrackerP1PXECHalfCylinder,111111",
#             "TrackerTIBHalfBarrel,111111",
#             "TrackerTIDEndcap,111111",
#             "TrackerTECEndcap,111111",
#         )
#     )
# ] # end of process.AlignmentProducer.RunRangeSelection

# # To run simultaneous calibrations of the pixel Lorentz angle you need to
# # include the corresponding config fragment and configure the granularity and
# # IOVs (must be consistent with input LA/template/alignment IOVs) for it.
# # Note: There are different version of the LA record available in the global
# #       tag. Depending on the TTRHBuilder, one has to set a label to configure
# #       which of them is to be used. The default TTRHBuilder uses pixel
# #       templates which ignores the unlabelled LA record and uses only the one
# #       labelled "fromAlignment". This is also the default value in the
# #       integrated LA calibration. If you are using the generic CPE instead of
# #       the template CPE you have to use the following setting:
# #
# #       siPixelLA.lorentzAngleLabel = ""
#
# from Alignment.CommonAlignmentAlgorithm.SiPixelLorentzAngleCalibration_cff \
#     import SiPixelLorentzAngleCalibration as siPixelLA
# siPixelLA.LorentzAngleModuleGroups.Granularity = cms.VPSet()
# siPixelLA.LorentzAngleModuleGroups.RunRange = cms.vuint32(290550,
#                                                           295000,
#                                                           298100)
#
# siPixelLA.LorentzAngleModuleGroups.Granularity.extend([
#     cms.PSet(
#         levels = cms.PSet(
#             alignParams = cms.vstring(
#                 'TrackerP1PXBModule,,RINGLAYER'
#             ),
#             RINGLAYER = cms.PSet(
#                 pxbDetId  = cms.PSet(
#                     moduleRanges = cms.vint32(ring, ring),
#                     layerRanges = cms.vint32(layer, layer)
#                 )
#             )
#         )
#     )
#     for ring in xrange(1,9) # [1,8]
#     for layer in xrange(1,5) # [1,4]
# ])
# siPixelLA.LorentzAngleModuleGroups.Granularity.append(
#     cms.PSet(
#         levels = cms.PSet(
#             alignParams = cms.vstring('TrackerP1PXECModule,,posz'),
#             posz = cms.PSet(zRanges = cms.vdouble(-9999.0, 9999.0))
#         )
#     )
# )
#
# process.AlignmentProducer.calibrations.append(siPixelLA)


#########################
## insert Pedesettings ##
#########################

# # reasonable pede settings are already defined in
# # 'confAliProducer.setConfiguration' above
# #
# # if you want to obtain alignment errors, use the following setting:
# # process.AlignmentProducer.algoConfig.pedeSteerer.method = "inversion 3 0.8"
# #
# # a list of possible options is documented here:
# # http://www.desy.de/~kleinwrt/MP2/doc/html/option_page.html#sec-cmd
# #
# # if you need to request a larger stack size for individual threads when
# # running pede, you can do this with this setting: 
# # process.AlignmentProducer.algoConfig.pedeSteerer.pedeCommand = "export OMP_STACKSIZE=20M; pede"
# #
# # you can change or drop pede options as follows:
#
# import Alignment.MillePedeAlignmentAlgorithm.alignmentsetup.helper as helper
# helper.set_pede_option(process, "entries 50 10 2")
# helper.set_pede_option(process, "compress", drop = True)


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
