import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('reRECO',eras.Run2_2018)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('FWCore.ParameterSet.Types')
from RecoPPS.Configuration.RecoPPS_EventContent_cff import RecoPPSAOD

import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing ()
options.register('nameTag',
                '',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.string,
                "String used to identify all files")
options.register('outputFileName',
                '',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.string,
                "output ROOT file name")
options.register('sourceFileName',
                '',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.string,
                "source file list name")
options.register('jsonFileName',
                '',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.string,
                "JSON file list name")
options.register('maxEvents',
                '',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.int,
                "Maximum number of events to analyze")
options.register('skipEvents',
                '',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.int,
                "Events to be skipped")
options.parseArguments()

process.options = cms.untracked.PSet(
    SkipEvent = cms.untracked.vstring('ProductNotFound')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents),
)

process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(1000)

import FWCore.Utilities.FileUtils as FileUtils

if options.sourceFileName == '':
    sourceFileName = 'InputFiles/'+options.nameTag+'.dat'
else: 
    sourceFileName = options.sourceFileName

print("Source file list: "+sourceFileName)
fileList = FileUtils.loadListFromFile(sourceFileName) 
inputFiles = cms.untracked.vstring(*fileList)

process.source = cms.Source("PoolSource",
    fileNames = inputFiles,
    duplicateCheckMode = cms.untracked.string("checkAllFilesOpened"),
    skipBadFiles = cms.untracked.bool(True),
    skipEvents = cms.untracked.uint32(options.skipEvents),
)

import FWCore.PythonUtilities.LumiList as LumiList

if options.jsonFileName == '':
    jsonFileName = 'JSONFiles/'+options.nameTag+'.json'
else:
    jsonFileName = options.jsonFileName

print("JSON file: "+jsonFileName)

if options.outputFileName == '':
    outputFileName = options.nameTag+'.root'
else:
    outputFileName = options.outputFileName

print("Output file: "+outputFileName)

from HLTrigger.HLTfilters.hltHighLevel_cfi import *
process.triggerSelection  = hltHighLevel.clone(TriggerResultsTag = "TriggerResults::HLT", HLTPaths = ['HLT_Ele32_WPTight_Gsf_L1DoubleEG_v*','HLT_Ele35_WPTight_Gsf_v*'])

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string(outputFileName),
    outputCommands = cms.untracked.vstring("drop *")
)
process.output.outputCommands.extend(RecoPPSAOD.outputCommands)

process.source.lumisToProcess = LumiList.LumiList(filename = jsonFileName).getVLuminosityBlockRange()

process.triggerSelection_step = cms.Path(process.triggerSelection)
process.output_step = cms.EndPath(process.output)

process.schedule = cms.Schedule(process.triggerSelection_step, process.output_step)