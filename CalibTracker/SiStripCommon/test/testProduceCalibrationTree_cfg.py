from __future__ import print_function
import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patInputFiles_cff import filesRelValTTbarPileUpGENSIMRECO
import FWCore.ParameterSet.VarParsing as VarParsing
from CalibTracker.SiStripCommon.shallowTree_test_template import *

###################################################################
def alterTriggersForUnitTest(process):
###################################################################
    '''
    These modification as necessary only in order to run on a
    RelVal MC in which the physics menu is not simulated!
    '''
    if(hasattr(process,'AAGFilter')):
        process.AAGFilter.triggerConditions = ["HLT_Random_v*"]
    if(hasattr(process,'IsolatedMuonFilter')):
        process.IsolatedMuonFilter.triggerConditions = ["MC_IsoMu_v*"]

    return process

###################################################################
# Setup 'standard' options
###################################################################
options = VarParsing.VarParsing()

options.register('conditionGT',
                 "auto:phase1_2017_realistic",
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "condition global tag for the job (\"auto:phase1_2017_realistic\" is default)")

options.register('conditionOverwrite',
                 "",
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "configuration to overwrite the condition into the GT (\"\" is default)")

options.register('inputCollection',
                 "generalTracks",
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "collections to be used for input (\"generalTracks\" is default)")

options.register('inputFiles',
                 filesRelValTTbarPileUpGENSIMRECO,
                 VarParsing.VarParsing.multiplicity.list,
                 VarParsing.VarParsing.varType.string,
                 "file to process")

options.register('outputFile',
                 "calibTreeTest.root",
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "name for the output root file (\"calibTreeTest.root\" is default)")

options.register('maxEvents',
                 -1,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "number of events to process (\"-1\" for all)")

options.register('unitTest',
                 False,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "is for unit test?")

options.parseArguments()

print("conditionGT       : ", options.conditionGT)
print("conditionOverwrite: ", options.conditionOverwrite)
print("inputCollection   : ", options.inputCollection)
print("maxEvents         : ", options.maxEvents)
print("outputFile        : ", options.outputFile)
print("inputFiles        : ", options.inputFiles)

process = cms.Process('CALIB')
#from CalibTracker.SiStripChannelGain.ntuple_cff import *
process.load('CalibTracker.Configuration.setupCalibrationTree_cff')
process.load('CalibTracker.SiStripCommon.ShallowEventDataProducer_cfi') #event Info

process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.Geometry.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('CalibTracker.SiStripCommon.ShallowEventDataProducer_cfi') #event Info

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.conditionGT, options.conditionOverwrite)

process.load('FWCore.MessageService.MessageLogger_cfi')
if(options.unitTest):
    process.MessageLogger.cerr.FwkReport.reportEvery = 1
else:
    process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.load('Configuration.StandardSequences.Services_cff')
process.add_( cms.Service( "TFileService",
                           fileName = cms.string( options.outputFile ),
                           closeFileFast = cms.untracked.bool(True)  ) )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.maxEvents) )

#import runs
process.source = cms.Source (
    "PoolSource",
    fileNames = cms.untracked.vstring(options.inputFiles[0])
    )

#definition of input collection
process.CalibrationTracks.src = cms.InputTag( options.inputCollection )
process.shallowTracks.Tracks  = cms.InputTag( options.inputCollection )
#process.shallowGainCalibrationAllBunch   = 'ALCARECOSiStripCalMinBias' #cms.InputTag( options.inputCollection )
#process.shallowGainCalibrationAllBunch0T = 'ALCARECOSiStripCalMinBias' #cms.InputTag( options.inputCollection )

# BSCNoBeamHalo selection (Not to use for Cosmic Runs) --- OUTDATED!!!
## process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
## process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')

## process.L1T1=process.hltLevel1GTSeed.clone()
## process.L1T1.L1TechTriggerSeeding = cms.bool(True)
## process.L1T1.L1SeedsLogicalExpression = cms.string('(40 OR 41) AND NOT (36 OR 37 OR 38 OR 39)')

compressionSettings = 201
process.EventInfo = cms.EDAnalyzer("ShallowTree", 
					CompressionSettings = process.gainCalibrationTreeStdBunch.CompressionSettings,
                            		outputCommands = cms.untracked.vstring('drop *',
                                                                          'keep *_shallowEventRun_*_*',
                                                                          )
                                   )

process.gainCalibrationTreeStdBunch.CompressionSettings = cms.untracked.int32(compressionSettings)
process.gainCalibrationTreeStdBunch0T.CompressionSettings = cms.untracked.int32(compressionSettings)
process.gainCalibrationTreeIsoMuon.CompressionSettings = cms.untracked.int32(compressionSettings)
process.gainCalibrationTreeIsoMuon0T.CompressionSettings = cms.untracked.int32(compressionSettings)
process.gainCalibrationTreeAagBunch.CompressionSettings = cms.untracked.int32(compressionSettings)
process.gainCalibrationTreeAagBunch0T.CompressionSettings = cms.untracked.int32(compressionSettings)

#process.TkCalPath = cms.Path(process.L1T1*process.TkCalFullSequence)

### if it is a unit test run, do not look for not existing triggers
### e.g. no AAG in MC

if(options.unitTest):
    alterTriggersForUnitTest(process)

process.TkCalPath_StdBunch   = cms.Path(process.TkCalSeq_StdBunch*process.shallowEventRun*process.EventInfo)
process.TkCalPath_StdBunch0T = cms.Path(process.TkCalSeq_StdBunch0T*process.shallowEventRun*process.EventInfo)
process.TkCalPath_IsoMuon    = cms.Path(process.TkCalSeq_IsoMuon*process.shallowEventRun*process.EventInfo)
process.TkCalPath_IsoMuon0T  = cms.Path(process.TkCalSeq_IsoMuon0T*process.shallowEventRun*process.EventInfo)
process.TkCalPath_AagBunch   = cms.Path(process.TkCalSeq_AagBunch*process.shallowEventRun*process.EventInfo)
process.TkCalPath_AagBunch0T = cms.Path(process.TkCalSeq_AagBunch0T*process.shallowEventRun*process.EventInfo)

process.schedule = cms.Schedule( process.TkCalPath_StdBunch, 
                                 process.TkCalPath_StdBunch0T,
                                 process.TkCalPath_IsoMuon,
                                 process.TkCalPath_IsoMuon0T,
                                 process.TkCalPath_AagBunch,
                                 process.TkCalPath_AagBunch0T,
                                 )

process.options = cms.untracked.PSet(
    #wantSummary = cms.untracked.bool(True),
    Rethrow = cms.untracked.vstring('OtherCMS', 
        'StdException', 
        'Unknown', 
        'BadAlloc', 
        'BadExceptionType', 
        'ProductNotFound', 
        'DictionaryNotFound', 
        'InsertFailure', 
        'Configuration', 
        'LogicError', 
        'UnimplementedFeature', 
        'InvalidReference', 
        'NullPointerError', 
        'NoProductSpecified', 
        'EventTimeout', 
        'EventCorruption', 
        'ScheduleExecutionFailure', 
        'EventProcessorFailure', 
        'FileInPathError', 
        'FileOpenError', 
        'FileReadError', 
        'FatalRootError', 
        'MismatchedInputFiles', 
        'ProductDoesNotSupportViews', 
        'ProductDoesNotSupportPtr', 
        'NotFound')
    )

process.Timing = cms.Service("Timing",
                             summaryOnly = cms.untracked.bool(True),
                             useJobReport = cms.untracked.bool(True)
                             )

'''
if(options.unitTest):
    process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
                                            ignoreTotal = cms.untracked.int32(1),
                                            moduleMemorySummary = cms.untracked.bool(True)
                                            )
'''

#Setup FWK for multithreaded
process.options.numberOfThreads=cms.untracked.uint32(4)
process.options.numberOfStreams=cms.untracked.uint32(0)
#process.options.numberOfConcurrentLuminosityBlocks=cms.untracked.uint32(2)  ## not yet
