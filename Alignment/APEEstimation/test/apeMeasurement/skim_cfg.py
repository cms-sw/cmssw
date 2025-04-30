import os
import FWCore.ParameterSet.Config as cms

##
## Process definition
##
process = cms.Process("ApeSkim")

##
## Input arguments
##
import FWCore.ParameterSet.VarParsing as VarParsing
import sys
options = VarParsing.VarParsing ('standard')
options.register('fileList', None, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, "File list name")
options.register('outputName', None, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, "Output file name")
options.register('trackSelection', "MinBias", VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, "Track selection")
options.register('globalTag', None, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, "Global Tag")
# ~ options.register('maxEvents', -1, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int, "Max number of events to be processed")
options.register('maxFileSize', 350000, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int, "Max number of events to be processed")

# get and parse the command line arguments
options.parseArguments()



##
## Message Logger
##
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.AlignmentTrackSelector=dict()
process.MessageLogger.cerr.INFO.limit = 0
process.MessageLogger.cerr.default.limit = -1
process.MessageLogger.cerr.AlignmentTrackSelector = cms.untracked.PSet(limit = cms.untracked.int32(-1))
process.MessageLogger.cerr.FwkReport.reportEvery = 1000 ## really show only every 1000th

##
## Process options
##
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True),
)


##
## Start of Configuration
##

outputName = "{}.root".format(options.outputName)
outputFileSize = options.maxFileSize
trackSelection = options.trackSelection
globalTag = options.globalTag
maxEvents = options.maxEvents

##
## Choice of GlobalTag
##
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, globalTag, '')
print("Using global tag "+process.GlobalTag.globaltag._value)

process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.Geometry.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")

import importlib
conditions = importlib.import_module("Alignment.APEEstimation.conditions.dataset_{}_cff".format(options.outputName))
conditions.applyConditions(process)

path, fn = os.path.split(options.fileList) 
import sys
sys.path.append(path)
fileList = importlib.import_module(fn.split(".")[0]) # remove .py ending from filename
process.source = fileList.source


##
## Number of Events (should be after input file)
##
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(maxEvents) )

##
## Skim tracks
##
import Alignment.APEEstimation.AlignmentTrackSelector_cff as AlignmentTrackSelector

# Determination of which AlignmentTrackSelector to use
if trackSelection in ["SingleMu", "SingleMuon"]:
    trackSelector = AlignmentTrackSelector.MuSkimSelector
elif trackSelection == "GenSim":
    trackSelector = AlignmentTrackSelector.genSimSkimSelector    
elif trackSelection in ["DoubleMu", "DoubleMuon"]:
    trackSelector = AlignmentTrackSelector.DoubleMuSkimSelector
elif trackSelection in  ["MinBias", "MinimumBias"]:
    trackSelector = AlignmentTrackSelector.MinBiasSkimSelector
elif trackSelection == "Cosmics":
    trackSelector = AlignmentTrackSelector.CosmicsSkimSelector
else: # Extend list here with custom track selectors
    print("Unknown trackSelection %s, exiting"%(trackSelection))
    exit(1)

process.MuSkim = trackSelector

import Alignment.CommonAlignment.tools.trackselectionRefitting as trackselRefit
process.seqTrackselRefit = trackselRefit.getSequence(process, trackSelector.src.getModuleLabel())


##
## Path
##
process.path = cms.Path(
    process.offlineBeamSpot*
    process.seqTrackselRefit*
    process.MuSkim
)

##
## Define event selection from path
##
EventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('path')
    )
)


##
## configure output module
##
process.out = cms.OutputModule("PoolOutputModule",
    ## Parameters directly for PoolOutputModule
    fileName = cms.untracked.string(outputName),
    # Maximum size per file before a new one is created
    maxSize = cms.untracked.int32(outputFileSize),
    dropMetaData = cms.untracked.string("DROPPED"),
    ## Parameters for inherited OutputModule
    SelectEvents = EventSelection.SelectEvents,
    outputCommands = cms.untracked.vstring(
        'drop *',
    ),
)
process.load("Alignment.APEEstimation.PrivateSkim_EventContent_cff")
process.out.outputCommands.extend(process.ApeSkimEventContent.outputCommands)


##
## Outpath
##
process.outpath = cms.EndPath(process.out)
-- dummy change --
