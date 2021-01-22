from __future__ import print_function
#
# cfg file to run the L1 GT test analyzer according to 
#   the options set in "user choices"
#

import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("L1GtAnalyzer")

print('\n')
from L1Trigger.GlobalTriggerAnalyzer.UserOptions_cff import *
if errorUserOptions == True :
    print('\nError returned by UserOptions_cff. Script stops here.\n')
    sys.exit()


# source according to data type
if dataType == 'StreamFile' :
    process.source = cms.Source("NewEventStreamFileReader", 
                                fileNames=readFiles,
                                lumisToProcess = selectedLumis,
                                eventsToProcess = selectedEvents
                                )
else :        
    process.source = cms.Source ('PoolSource', 
                                 fileNames=readFiles, 
                                 secondaryFileNames=secFiles,
                                 lumisToProcess = selectedLumis,
                                 eventsToProcess = selectedEvents
                                 )


# number of events to be processed and source file
process.maxEvents = cms.untracked.PSet(
    input=cms.untracked.int32(maxNumberEvents)
)

#
# load and configure modules via Global Tag
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions

process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.GlobalTag.globaltag = useGlobalTag

# processes to be run


process.load("L1Trigger.GlobalTriggerAnalyzer.L1GtAnalyzer_cff")

# print output: 0 = std::cout; 1 = LogTrace; 2 = LogVerbatim; 3 = LogInfo
process.l1GtAnalyzer.PrintOutput = 3

# enable / disable various analysis methods
#process.l1GtAnalyzer.analyzeDecisionReadoutRecordEnable = True
#
#process.l1GtAnalyzer.analyzeL1GtUtilsMenuLiteEnable = True
process.l1GtAnalyzer.analyzeL1GtUtilsEventSetupEnable = True
#process.l1GtAnalyzer.analyzeL1GtUtilsEnable = True
#process.l1GtAnalyzer.analyzeTriggerEnable = True
#
#process.l1GtAnalyzer.analyzeObjectMapEnable = True
#
#process.l1GtAnalyzer.analyzeL1GtTriggerMenuLiteEnable = True
#
#process.l1GtAnalyzer.analyzeConditionsInRunBlockEnable = True
#process.l1GtAnalyzer.analyzeConditionsInLumiBlockEnable = True
#process.l1GtAnalyzer.analyzeConditionsInEventBlockEnable = True

# 
#
# input tag for GT readout collection: 
#process.l1GtAnalyzer.L1GtDaqInputTag = 'gtDigis' 
 
# input tags for GT lite record
#process.l1GtAnalyzer.L1GtRecordInputTag = 'l1GtRecord'

# input tag for GT object map collection
#process.l1GtAnalyzer.L1GtObjectMapTag = 'hltL1GtObjectMap'

# input tag for L1GtTriggerMenuLite
#process.l1GtAnalyzer.L1GtTmLInputTag = 'l1GtTriggerMenuLite'

# input tag for ConditionInEdm products
#process.l1GtAnalyzer.CondInEdmInputTag = 'conditionsInEdm'

# physics algorithm name or alias, technical trigger name 
#process.l1GtAnalyzer.AlgorithmName = 'L1_SingleEG20'
#process.l1GtAnalyzer.AlgorithmName = 'L1_DoubleMu0er_HighQ'
process.l1GtAnalyzer.AlgorithmName = 'L1_SingleMu14er'
#process.l1GtAnalyzer.AlgorithmName = 'L1_BscMinBiasOR_BptxPlusORMinus'
#process.l1GtAnalyzer.AlgorithmName = 'L1Tech_BPTX_plus_AND_minus_instance1.v0'
#process.l1GtAnalyzer.AlgorithmName = 'L1Tech_BPTX_quiet.v0'
#process.l1GtAnalyzer.AlgorithmName = 'L1Tech_BPTX_plus_AND_minus.v0'

# condition in the above algorithm to test the object maps
#process.l1GtAnalyzer.ConditionName = 'SingleIsoEG_0x14'
#process.l1GtAnalyzer.ConditionName = 'DoubleMu_0x01_HighQ_EtaCuts'
process.l1GtAnalyzer.ConditionName = 'DoubleMu_0x01_HighQ_EtaCuts'

# a bit number
process.l1GtAnalyzer.BitNumber = 10

# select the L1 configuration use: 0 (default), 100000, 200000
#process.l1GtAnalyzer.L1GtUtilsConfiguration = 0
#process.l1GtAnalyzer.L1GtUtilsConfiguration = 100000
process.l1GtAnalyzer.L1GtUtilsConfiguration = 200000
 
# if true, use methods in L1GtUtils with the input tag for L1GtTriggerMenuLite
# from provenance (default: True)
#process.l1GtAnalyzer.L1GtTmLInputTagProv = False

# if true, configure (partially) L1GtUtils in beginRun using getL1GtRunCache
# (default: True)
process.l1GtAnalyzer.L1GtUtilsConfigureBeginRun = True


process.load("L1Trigger.GlobalTriggerAnalyzer.l1GtTrigReport_cfi")

# boolean flag to select the input record
#process.l1GtTrigReport.UseL1GlobalTriggerRecord = True

# input tag for the GT record requested: 
#   GT emulator:    gtDigis (DAQ record)
#   GT unpacker:    gtDigis (DAQ record)
#   GT lite record: l1GtRecord 
#process.l1GtTrigReport.L1GtRecordInputTag = "gtDigis"

process.l1GtTrigReport.PrintVerbosity = 10

# print output: 0 = std::cout; 1 = LogTrace; 2 = LogVerbatim; 3 = LogInfo
process.l1GtTrigReport.PrintOutput = 3


# for RAW data, run first the RAWTODIGI and then L1Reco
if ((dataType == 'RAW') or (dataType == 'StreamFile')) and not (useRelValSample) :
    process.load('Configuration/StandardSequences/RawToDigi_Data_cff')
    process.load('L1Trigger/Configuration/L1Reco_cff')
    process.l1GtTrigReport.L1GtRecordInputTag = "gtDigis"
    process.p = cms.Path(process.RawToDigi+process.L1Reco+process.l1GtTrigReport+process.l1GtAnalyzer)

elif (dataType == 'RAW') and (useRelValSample) :
    process.load('Configuration/StandardSequences/RawToDigi_cff')
    process.load('L1Trigger/Configuration/L1Reco_cff')
    process.l1GtTrigReport.L1GtRecordInputTag = "gtDigis"
    process.p = cms.Path(process.RawToDigi+process.L1Reco+process.l1GtTrigReport+process.l1GtAnalyzer)
    
else :        
    # path to be run for RECO and AOD
    process.p = cms.Path(process.l1GtTrigReport+process.l1GtAnalyzer)


# Message Logger
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.debugModules = ['l1GtAnalyzer']

process.MessageLogger.cerr.enable = False
process.MessageLogger.cerr.default.limit = 0
process.MessageLogger.cerr.FwkReport.limit = 0
process.MessageLogger.cerr.FwkSummary.limit = 0

process.MessageLogger.files.L1GtAnalyzer_debug = cms.untracked.PSet( 
        threshold = cms.untracked.string('DEBUG'),
        DEBUG = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        INFO = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        WARNING = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        L1GtAnalyzer = cms.untracked.PSet( limit = cms.untracked.int32(0) ), 
        L1GtUtils = cms.untracked.PSet( limit = cms.untracked.int32(-1) ), 
        L1GtTrigReport = cms.untracked.PSet( limit = cms.untracked.int32(0) ) 
        )

process.MessageLogger.files.L1GtAnalyzer_info = cms.untracked.PSet( 
        threshold = cms.untracked.string('INFO'),
        INFO = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        WARNING = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        L1GtAnalyzer = cms.untracked.PSet( limit = cms.untracked.int32(-1) ), 
        L1GtUtils = cms.untracked.PSet( limit = cms.untracked.int32(-1) ), 
        L1GtTrigReport = cms.untracked.PSet( limit = cms.untracked.int32(-1) ) 
        )

process.MessageLogger.files.L1GtAnalyzer_warning = cms.untracked.PSet( 
        threshold = cms.untracked.string('WARNING'),
        WARNING = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        L1GtAnalyzer = cms.untracked.PSet( limit = cms.untracked.int32(-1) ), 
        L1GtUtils = cms.untracked.PSet( limit = cms.untracked.int32(-1) ), 
        L1GtTrigReport = cms.untracked.PSet( limit = cms.untracked.int32(-1) ) 
        )

process.MessageLogger.files.L1GtAnalyzer_error = cms.untracked.PSet( 
        threshold = cms.untracked.string('ERROR'),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
        L1GtAnalyzer = cms.untracked.PSet( limit = cms.untracked.int32(-1) ), 
        L1GtUtils = cms.untracked.PSet( limit = cms.untracked.int32(-1) ), 
        L1GtTrigReport = cms.untracked.PSet( limit = cms.untracked.int32(-1) ) 
       )
