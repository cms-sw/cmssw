#
# cfg file to run L1GtTrigReport on GT output file containing 
#    the readout record L1GlobalTriggerReadoutRecord
#    or
#    the lite record L1GlobalTriggerRecord
#
# V M Ghete 2009-03-04


import FWCore.ParameterSet.Config as cms
import sys

# process
process = cms.Process('L1GtTrigReport')

print '\n'
from L1Trigger.GlobalTriggerAnalyzer.UserOptions_cff import *
if errorUserOptions == True :
    print '\nError returned by UserOptions_cff\n'
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

# load and configure modules via Global Tag
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions

process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.GlobalTag.globaltag = useGlobalTag+'::All'

#
# l1GtTrigReport module
#

process.load("L1Trigger.GlobalTriggerAnalyzer.l1GtTrigReport_cfi")
 
# boolean flag to select the input record
# if true, it will use L1GlobalTriggerRecord 
#process.l1GtTrigReport.UseL1GlobalTriggerRecord = True

# input tag for GT record: 
#   GT emulator:    gtDigis (DAQ record)
#   GT unpacker:    gtDigis (DAQ record)
#   GT lite record: l1GtRecord 
process.l1GtTrigReport.L1GtRecordInputTag = "simGtDigis"

#process.l1GtTrigReport.PrintVerbosity = 2
#process.l1GtTrigReport.PrintOutput = 1

# for RAW data, run first the RAWTODIGI 
if (dataType == 'RAW') and not (useRelValSample) :
    process.load('Configuration/StandardSequences/RawToDigi_Data_cff')
    process.l1GtTrigReport.L1GtRecordInputTag = "gtDigis"
    process.p = cms.Path(process.RawToDigi+process.l1GtTrigReport)

elif (dataType == 'RAW') and (useRelValSample) :
    process.load('Configuration/StandardSequences/RawToDigi_cff')
    process.l1GtTrigReport.L1GtRecordInputTag = "gtDigis"
    process.p = cms.Path(process.RawToDigi+process.l1GtTrigReport)
    
else :        
    # path to be run for RECO
    process.p = cms.Path(process.l1GtTrigReport)

# Message Logger
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.debugModules = ['l1GtTrigReport']
process.MessageLogger.categories.append('L1GtTrigReport')

#process.MessageLogger.cerr.threshold = 'DEBUG'
process.MessageLogger.cerr.threshold = 'INFO'
#process.MessageLogger.cerr.threshold = 'WARNING'
#process.MessageLogger.cerr.threshold = 'ERROR'

process.MessageLogger.cerr.DEBUG = cms.untracked.PSet( limit = cms.untracked.int32(0) )
process.MessageLogger.cerr.INFO = cms.untracked.PSet( limit = cms.untracked.int32(0) )
process.MessageLogger.cerr.WARNING = cms.untracked.PSet( limit = cms.untracked.int32(0) )
process.MessageLogger.cerr.ERROR = cms.untracked.PSet( limit = cms.untracked.int32(0) )

process.MessageLogger.cerr.L1GtTrigReport = cms.untracked.PSet( limit = cms.untracked.int32(-1) )


# summary
process.options = cms.untracked.PSet(
    wantSummary=cms.untracked.bool(True)
)

