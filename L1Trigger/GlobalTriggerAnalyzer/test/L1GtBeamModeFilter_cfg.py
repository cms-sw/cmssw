from __future__ import print_function
#
# cfg file to test L1GtBeamModeFilter
# it requires as input:
#   a RAW data-tier sample (with FED 812 included) or 
#   a digi data-tier sample, with L1GlobalTriggerEvmReadoutRecord product valid or
#   a RECO data-tier sample, with ConditionsInEdm product valid
#
# the input sample is chosen in L1Trigger.GlobalTriggerAnalyzer.UserOptions_cff
#
# V.M. Ghete 2010-06-22
#

import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process('TestL1GtBeamModeFilter')

print('\n')
from L1Trigger.GlobalTriggerAnalyzer.UserOptions_cff import *
if errorUserOptions == True :
    print('\nError returned by UserOptions_cff\n')
    sys.exit()


# source according to data type
if dataType == 'StreamFile' :
    process.source = cms.Source("NewEventStreamFileReader", fileNames=readFiles)
else :        
    process.source = cms.Source ('PoolSource', fileNames=readFiles, secondaryFileNames=secFiles)


# number of events to be processed and source file
process.maxEvents = cms.untracked.PSet(
    input=cms.untracked.int32(maxNumberEvents)
)

#
# load and configure modules via Global Tag

process.load("Configuration.StandardSequences.GeometryDB_cff")
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.GlobalTag.globaltag = useGlobalTag+'::All'





# this module
process.load('L1Trigger.GlobalTriggerAnalyzer.l1GtBeamModeFilter_cfi')
 
# replacing arguments for l1GtBeamModeFilter

# input tag for input tag for ConditionInEdm products
#process.l1GtBeamModeFilter.CondInEdmInputTag = cms.InputTag("conditionsInEdm"),

# input tag for the L1 GT EVM product 
#process.l1GtBeamModeFilter.L1GtEvmReadoutRecordTag = cms.InputTag("gtEvmDigis")

# vector of allowed beam modes (see enumeration in header file for implemented values)
# default value: 11 (STABLE)
#process.l1GtBeamModeFilter.AllowedBeamMode = cms.vuint32(11)
process.l1GtBeamModeFilter.AllowedBeamMode = cms.vuint32(9, 10, 11)

# return the inverted result, to be used instead of NOT
#process.l1GtBeamModeFilter.InvertResult = False

# path to be run
# for RAW data, run first the RAWTODIGI 
if (dataType == 'RAW') and not (useRelValSample) :
    process.load('Configuration/StandardSequences/RawToDigi_Data_cff')
    process.gtEvmDigis.UnpackBxInEvent = cms.int32(1)
    # set EVM unpacker to verbose
    process.gtEvmDigis.Verbosity = cms.untracked.int32(1)

    process.p = cms.Path(process.RawToDigi * process.l1GtBeamModeFilter)

elif (dataType == 'RAW') and (useRelValSample) :
    process.load('Configuration/StandardSequences/RawToDigi_cff')
    process.gtEvmDigis.UnpackBxInEvent = cms.int32(1)
    # set EVM unpacker to verbose
    process.gtEvmDigis.Verbosity = cms.untracked.int32(1)

    process.p = cms.Path(process.RawToDigi * process.l1GtBeamModeFilter)
    
else :        
    # run on standard RECO
    process.p = cms.Path(process.l1GtBeamModeFilter)

# services

# Message Logger
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.debugModules = ['gtEvmDigis', 'l1GtBeamModeFilter']
process.MessageLogger.L1GlobalTriggerEvmRawToDigi=dict()
process.MessageLogger.L1GtBeamModeFilter=dict()

process.MessageLogger.cerr.threshold = 'DEBUG'
#process.MessageLogger.cerr.threshold = 'INFO'
#process.MessageLogger.cerr.threshold = 'WARNING'
#process.MessageLogger.cerr.threshold = 'ERROR'

process.MessageLogger.cerr.DEBUG = cms.untracked.PSet( limit = cms.untracked.int32(0) )
process.MessageLogger.cerr.INFO = cms.untracked.PSet( limit = cms.untracked.int32(0) )
process.MessageLogger.cerr.WARNING = cms.untracked.PSet( limit = cms.untracked.int32(0) )
process.MessageLogger.cerr.ERROR = cms.untracked.PSet( limit = cms.untracked.int32(0) )

process.MessageLogger.cerr.L1GlobalTriggerEvmRawToDigi = cms.untracked.PSet( limit = cms.untracked.int32(0) )
process.MessageLogger.cerr.L1GtBeamModeFilter = cms.untracked.PSet( limit = cms.untracked.int32(-1) )


