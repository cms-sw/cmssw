####### Universal configuration template for tracker alignment
##
##  Usage:
##    
##    Make a copy of this file and insert Startgeometry, Alignables and 
##    Pedesettings directly into it.
##
##    Specify the path to this config-Template in the alignment_setup.ini
##
##    The scripts mps_alisetup.py and mps_setup.py set the Variables at the top (setup*).
##    
##        Collection specifies the type of Tracks. Currently these are supported:
##          - ALCARECOTkAlMinBias       -> Minimum Bias
##          - ALCARECOTkAlCosmicsCTF0T  -> Cosmics, either at 0T or 3.8T
##          - ALCARECOTkAlMuonIsolated  -> Isolated Muon
##          - ALCARECOTkAlZMuMu         -> Z decay to two Muons
##
##        Globaltag specifies the detector conditions.
##        Parts of the Globaltag are overwritten in Startgeometry.
##
##        monitorFile and binaryFile are automatically set by mps_setup.
##        e.g. millePedeMonitor004.root and milleBinary004.dat
##
##        AlgoMode specifies mode of AlignmentProducer.algoConfig -> mille or pede
##        mille is default. Pede mode is automatically set when merge config is created by MPS
##
##        CosmicsDecoMode and CosmicsZeroTesla are only relevant if collection 
##        is ALCARECOTkAlCosmicsCTF0T
##
##        If primaryWidth is bigger than 0.0 it overwrites 
##        process.AlignmentProducer.algoConfig.TrajectoryFactory.ParticleProperties.PrimaryWidth = ...
##        if primaryWidth<=0.0 it has no effect at all.

import FWCore.ParameterSet.Config as cms
process = cms.Process("Alignment")


## Variables edited by mps_alisetup.py. Used in functions below.
## You can change them manually as well.
## -----------------------------------------------------------------------------
setupGlobaltag        = "placeholder_globaltag"
setupCollection       = "placeholder_collection"
setupCosmicsDecoMode  = False
setupCosmicsZeroTesla = False
setupPrimaryWidth     = -1.0
setupJson             = "placeholder_json"

## Variables edited by MPS (mps_setup and mps_merge). Be careful.
## -----------------------------------------------------------------------------
# Default is "mille". Gets changed to "pede" by mps_merge.
setupAlgoMode         = "mille"

# MPS looks specifically for the string "ISN" so don't change this.
setupMonitorFile      = "millePedeMonitorISN.root"
setupBinaryFile       = "milleBinaryISN.dat"

# Input files. Edited by mps_splice.py
readFiles = cms.untracked.vstring()



## MessageLogger for convenient output
## -----------------------------------------------------------------------------
process.load('Alignment.MillePedeAlignmentAlgorithm.alignmentsetup.myMessageLogger_cff')


## Load the conditions
## -----------------------------------------------------------------------------
if setupCosmicsZeroTesla:
    # actually only needed for 0T MC samples, but does not harm for 0T data
    process.load("Configuration.StandardSequences.MagneticField_0T_cff") # B-field map
else:
    process.load('Configuration.StandardSequences.MagneticField_cff') # B-field map
process.load('Configuration.Geometry.GeometryRecoDB_cff') # Ideal geometry and interface
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff") # Global tag
process.GlobalTag.connect   = "frontier://FrontierProd/CMS_CONDITIONS"
process.GlobalTag.globaltag = setupGlobaltag


## Overwrite some conditions in global tag
## -----------------------------------------------------------------------------
import Alignment.MillePedeAlignmentAlgorithm.alignmentsetup.SetCondition as tagwriter

##########################
## insert Startgeometry ##
##########################

##  You can use tagwriter.setCondition() to overwrite conditions in globaltag
##  Example:
##  tagwriter.setCondition(process,
##      connect = 'frontier://FrontierProd/CMS_CONDITIONS',
##      record  = 'TrackerAlignmentRcd',
##      tag     = 'TrackerAlignment_Run2015B_PseudoPCL_v2')


## Alignment producer
## -----------------------------------------------------------------------------
process.load("Alignment.CommonAlignmentProducer.AlignmentProducer_cff")

#######################
## insert Alignables ##
#######################

import Alignment.MillePedeAlignmentAlgorithm.alignmentsetup.ConfigureAlignmentProducer as confAliProducer

confAliProducer.setConfiguration(process,
    collection   = setupCollection,
    mode         = setupAlgoMode,
    monitorFile  = setupMonitorFile,
    binaryFile   = setupBinaryFile,
    primaryWidth = setupPrimaryWidth)


#########################
## insert Pedesettings ## 
#########################


## Mille-procedure
## -----------------------------------------------------------------------------
if setupAlgoMode == "mille":
    
    # no database output in the mille step
    process.AlignmentProducer.saveDeformationsToDB = False
    
    ## Track selection and refitting
    ## -----------------------------------------------------------------------------
    process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")
    
    import Alignment.CommonAlignment.tools.trackselectionRefitting as trackRefitter
    process.TrackRefittingSequence = trackRefitter.getSequence(
        process, 
        setupCollection, 
        cosmicsDecoMode = setupCosmicsDecoMode,
        cosmicsZeroTesla = setupCosmicsZeroTesla)
    
    ## Overwrite Track-Selector filter from unified Sequence to false 
    process.AlignmentTrackSelector.filter = False
    if setupCollection != "ALCARECOTkAlCosmicsCTF0T":
        # there is no HighPurity selector for cosmics
        process.HighPurityTrackSelector.filter = False
    
    
    ## Configure the input data.
    ## -----------------------------------------------------------------------------
    process.source = cms.Source(
        "PoolSource",
        skipEvents = cms.untracked.uint32(0),
        fileNames  = readFiles
    )
    
    # Set Luminosity-Blockrange from json-file if given
    if (setupJson != "") and (setupJson != "placeholder_json"):
        import FWCore.PythonUtilities.LumiList as LumiList
        process.source.lumisToProcess = LumiList.LumiList(filename = setupJson).getVLuminosityBlockRange() 
    
    
    ## The executed path
    ## -----------------------------------------------------------------------------
    process.p = cms.Path(
        process.offlineBeamSpot
        *process.TrackRefittingSequence)


## Pede-procedure
## -----------------------------------------------------------------------------
else:
    
    ## Replace "save to DB" directives
    ## -----------------------------------------------------------------------------
    from CondCore.CondDB.CondDB_cfi import *
    process.PoolDBOutputService = cms.Service("PoolDBOutputService",
        CondDB,
        timetype = cms.untracked.string('runnumber'),
        toPut = cms.VPSet(cms.PSet(
            record = cms.string('TrackerAlignmentRcd'),
            tag = cms.string('Alignments')
        ),
            cms.PSet(
                record = cms.string('TrackerAlignmentErrorExtendedRcd'),
                tag = cms.string('AlignmentErrorsExtended')
            ),
            cms.PSet(
                record = cms.string('TrackerSurfaceDeformationRcd'),
                tag = cms.string('Deformations')
            ),
            cms.PSet(
                record = cms.string('SiStripLorentzAngleRcd_peak'),
                tag = cms.string('SiStripLorentzAngle_peak')
            ),
            cms.PSet(
                record = cms.string('SiStripLorentzAngleRcd_deco'),
                tag = cms.string('SiStripLorentzAngle_deco')
            ),
            cms.PSet(
                record = cms.string('SiPixelLorentzAngleRcd'),
                tag = cms.string('SiPixelLorentzAngle')
            ),
            cms.PSet(
                record = cms.string('SiStripBackPlaneCorrectionRcd'),
                tag = cms.string('SiStripBackPlaneCorrection')
            )
        )
    )    
    process.PoolDBOutputService.connect = 'sqlite_file:alignments_MP.db'
    process.AlignmentProducer.saveToDB = True
    
    
    ## Reconfigure parts of the algorithm configuration
    ## -----------------------------------------------------------------------------
    
    # placeholers get replaced by mps_merge.py, which is called in mps_setup.pl
    process.AlignmentProducer.algoConfig.mergeBinaryFiles = ['placeholder_binaryList']
    process.AlignmentProducer.algoConfig.mergeTreeFiles   = ['placeholder_treeList']
    
    
    ## Set a new source and path.
    ## -----------------------------------------------------------------------------
    process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1))
    process.source = cms.Source("EmptySource")
    process.dump = cms.EDAnalyzer("EventContentAnalyzer")
    process.p = cms.Path(process.dump)

