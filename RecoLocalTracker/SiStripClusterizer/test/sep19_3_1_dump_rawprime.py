import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
from Configuration.StandardSequences.Eras import eras

import os
import glob as glob

process = cms.Process('USER',eras.Run3_2023)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

options = VarParsing ('analysis')
options.register ('eventsToProcessTxt',
              '',
              VarParsing.multiplicity.singleton,
              VarParsing.varType.string,
              "Events to process text")
options.register ('n',
                  -1, # default value
                  VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.varType.int,          # string, int, bool or float
                  "n")
options.parseArguments()
print('inputFiles:\t', options.inputFiles)
print('eventsToProcessTxt:\t', options.eventsToProcessTxt)
print('outputFile:\t', options.outputFile)

if options.n != -1:
    process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(options.n)
    )

# process.source = cms.Source("PoolSource",
process.source = cms.Source(
    "NewEventStreamFileReader" if ".dat" in options.inputFiles[0] else "PoolSource",
    fileNames = cms.untracked.vstring(['file:'+f for f in options.inputFiles]),
    # secondaryFileNames = cms.untracked.vstring(),
    # eventsToProcess = cms.untracked.VEventRange( list( open(options.eventsToProcessTxt).readlines() ) )
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step2 nevts:30'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition
if (not os.path.exists('output/')): os.makedirs('output/')

process.TFileService = cms.Service("TFileService", fileName=cms.string(options.outputFile))

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '132X_dataRun3_HLT_v2', '')

process.sep19_3_dump_deadStrips = cms.EDAnalyzer("sep19_3_dump_deadStrips")

# Path and EndPath definitions
process.analyzer_step = cms.Path(process.sep19_3_dump_deadStrips) 
process.endjob_step = cms.EndPath(process.endOfProcess)

# Schedule definition
process.schedule = cms.Schedule(process.analyzer_step, process.endjob_step)
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

from Configuration.Applications.ConfigBuilder import MassReplaceInputTag
MassReplaceInputTag(process, new="rawDataMapperByLabel", old="rawDataCollector")

#do not add changes to your config after this point (unless you know what you are doing)
from FWCore.ParameterSet.Utilities import convertToUnscheduled
process=convertToUnscheduled(process)


# Customisation from command line
process.options.numberOfThreads = 8
process.options.numberOfStreams = 0

#Have logErrorHarvester wait for the same EDProducers to finish as those providing data for the OutputModule
from FWCore.Modules.logErrorHarvester_cff import customiseLogErrorHarvesterUsingOutputCommands
process = customiseLogErrorHarvesterUsingOutputCommands(process)

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion