import os

import FWCore.ParameterSet.Config as cms




##
## Setup command line options
##
import FWCore.ParameterSet.VarParsing as VarParsing
import sys
options = VarParsing.VarParsing ('standard')
options.register('workingArea', None, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, "Working area")
options.register('globalTag', "None", VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, "Custom global tag")
options.register('measName', None, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, "Folder in which to store results")
options.register('fileNumber', 1, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int, "Input file number")
options.register('iteration', 0, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int, "Iteration number")
options.register('lastIter', False, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.bool, "Last iteration")
options.register('isCosmics', False, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.bool, "Cosmic data set")
# get and parse the command line arguments
options.parseArguments()   

##
## Process definition
##
process = cms.Process("ApeEstimator")


process.load("Configuration.Geometry.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
from CondCore.CondDB.CondDB_cfi import *

##
## Message Logger
##
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.SectorBuilder=dict()
process.MessageLogger.ResidualErrorBinning=dict()
process.MessageLogger.HitSelector=dict()
process.MessageLogger.CalculateAPE=dict()
process.MessageLogger.ApeEstimator=dict()
process.MessageLogger.TrackRefitter=dict()
process.MessageLogger.AlignmentTrackSelector=dict()
process.MessageLogger.cerr.threshold = 'WARNING'
process.MessageLogger.cerr.INFO.limit = 0
process.MessageLogger.cerr.default.limit = -1
process.MessageLogger.cerr.SectorBuilder = cms.untracked.PSet(limit = cms.untracked.int32(-1))
process.MessageLogger.cerr.HitSelector = cms.untracked.PSet(limit = cms.untracked.int32(-1))
process.MessageLogger.cerr.CalculateAPE = cms.untracked.PSet(limit = cms.untracked.int32(-1))
process.MessageLogger.cerr.ApeEstimator = cms.untracked.PSet(limit = cms.untracked.int32(-1))
process.MessageLogger.cerr.AlignmentTrackSelector = cms.untracked.PSet(limit = cms.untracked.int32(-1))
process.MessageLogger.cerr.FwkReport.reportEvery = 1000 ## really show only every 1000th


##
## Process options
##
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True),
)


##
## Input Files
##
readFiles = cms.untracked.vstring()
process.source = cms.Source ("PoolSource",
    fileNames = readFiles
)
readFiles.extend( [
    'file:reco.root',
] )



##
## Number of Events (should be after input file)
##
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.maxEvents) ) # maxEvents is included in options by default



##
## Check run and event numbers for Dublicates --- only for real data
##
process.source.duplicateCheckMode = cms.untracked.string("checkEachRealDataFile")
#process.source.duplicateCheckMode = cms.untracked.string("checkAllFilesOpened")   # default value


##
## Whole Refitter Sequence
##
process.load("Alignment.APEEstimation.TrackRefitter_38T_cff")
process.GlobalTag = GlobalTag(process.GlobalTag, options.globalTag, '')



import importlib
mod = importlib.import_module("Alignment.APEEstimation.conditions.measurement_{}_cff".format(options.measName))
mod.applyConditions(process)


## APE
if options.iteration!=0:
    CondDBAlignmentError = CondDB.clone(connect = cms.string('sqlite_file:'+os.path.join(options.workingArea,options.measName)+'/apeObjects/apeIter'+str(options.iteration-1)+'.db'))
    process.myTrackerAlignmentErr = cms.ESSource("PoolDBESSource",
        CondDBAlignmentError,
        toGet = cms.VPSet(
            cms.PSet(
                record = cms.string('TrackerAlignmentErrorExtendedRcd'),
                tag = cms.string('APEs')
            )
        )
    )
    process.es_prefer_trackerAlignmentErr = cms.ESPrefer("PoolDBESSource","myTrackerAlignmentErr")


##
## ApeEstimator
##
from Alignment.APEEstimation.ApeEstimator_cff import *
process.ApeEstimator1 = ApeEstimator.clone(
    tjTkAssociationMapTag = "TrackRefitterForApeEstimator",
    applyTrackCuts = False,
    analyzerMode = False,
    calculateApe = True,
    Sectors = RecentSectors,
)

process.ApeEstimator2 = process.ApeEstimator1.clone(
  Sectors = ValidationSectors,
  analyzerMode = True,
  calculateApe = False,
)
process.ApeEstimator3 = process.ApeEstimator2.clone(
    zoomHists = False,
)

process.ApeEstimatorSequence = cms.Sequence(process.ApeEstimator1)
if options.iteration==0:
  process.ApeEstimatorSequence *= process.ApeEstimator2
  process.ApeEstimatorSequence *= process.ApeEstimator3
elif options.lastIter == True:
  process.ApeEstimatorSequence *= process.ApeEstimator2



##
## Output File Configuration
##
process.TFileService = cms.Service("TFileService",
    fileName = cms.string(os.path.join(options.workingArea,options.measName,"out"+str(options.fileNumber)+".root")),
    closeFileFast = cms.untracked.bool(True)
)



##
## Path
##

if not options.isCosmics:
    process.p = cms.Path(
        process.RefitterHighPuritySequence*
        process.ApeEstimatorSequence
    )
else:
    process.p = cms.Path(
        process.RefitterNoPuritySequence* # this sequence doesn't include high purity track criteria
        process.ApeEstimatorSequence
    )



-- dummy change --
