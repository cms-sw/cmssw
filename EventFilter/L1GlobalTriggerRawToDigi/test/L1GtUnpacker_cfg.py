from __future__ import print_function
#
# cfg file to unpack RAW L1 GT DAQ data
# the options set in "user choices" file
#   L1Trigger/GlobalTriggerAnalyzer/python/UserOptions.py
 
# V M Ghete 2009-04-03
# V M Ghete 2011-02-09 use UserOptions.py

 
import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("TestL1GtUnpacker")

print('\n')
from L1Trigger.GlobalTriggerAnalyzer.UserOptions_cff import *
if errorUserOptions == True :
    print('\nError returned by UserOptions_cff\n')
    sys.exit()

# source according to data type
if dataType == 'StreamFile' :
    process.source = cms.Source("NewEventStreamFileReader", fileNames=readFiles)
else :        
    process.source = cms.Source ('PoolSource', 
                                 fileNames=readFiles, 
                                 secondaryFileNames=secFiles,
                                 eventsToProcess = selectedEvents
                                 )


# number of events to be processed and source file
process.maxEvents = cms.untracked.PSet(
    input=cms.untracked.int32(maxNumberEvents)
)


# load and configure modules via Global Tag
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions

process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.GlobalTag.globaltag = useGlobalTag

# L1 GT/GMT unpack
process.load("EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi")

# input tag for GT readout collection (before CMSSW_5_0_X)
#     source        = hardware record
#
#if useRelValSample == True :
#    daqGtInputTag = 'rawDataCollector'
#else :
#    daqGtInputTag = 'rawDataCollector'

daqGtInputTag = 'rawDataCollector'

process.l1GtUnpack.DaqGtInputTag = daqGtInputTag
#process.l1GtUnpack.DaqGtInputTag = 'l1GtTextToRaw'

# Active Boards Mask

# no board masked (default)
#process.l1GtUnpack.ActiveBoardsMask = 0xFFFF
    
# GTFE only in the record
#process.l1GtUnpack.ActiveBoardsMask = 0x0000

# GTFE + FDL 
#process.l1GtUnpack.ActiveBoardsMask = 0x0001
     
# GTFE + GMT 
#process.l1GtUnpack.ActiveBoardsMask = 0x0100

# GTFE + FDL + GMT 
#process.l1GtUnpack.ActiveBoardsMask = 0x0101

# BxInEvent to be unpacked
# all available BxInEvent (default)
#process.l1GtUnpack.UnpackBxInEvent = -1 

# BxInEvent = 0 (L1A)
#process.l1GtUnpack.UnpackBxInEvent = 1 

# 3 BxInEvent (F, 0, 1)  
#process.l1GtUnpack.UnpackBxInEvent = 3 

# set it to verbose
process.l1GtUnpack.Verbosity = cms.untracked.int32(1)

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
process.l1GtTrigReport.L1GtRecordInputTag = "l1GtUnpack"

#process.l1GtTrigReport.PrintVerbosity = 10

# print output: 0 = std::cout; 1 = LogTrace; 2 = LogVerbatim; 3 = LogInfo
#process.l1GtTrigReport.PrintOutput = 0


# path to be run
process.p = cms.Path(process.l1GtUnpack*process.l1GtTrigReport)

# Message Logger
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.debugModules = ['l1GtUnpack', 'l1GtTrigReport']
process.MessageLogger.cerr.enable = False

process.MessageLogger.files.L1GtUnpacker_errors = cms.untracked.PSet( 
        threshold = cms.untracked.string('ERROR'),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
        L1GlobalTriggerRawToDigi = cms.untracked.PSet( limit = cms.untracked.int32(-1) ) 
       )

process.MessageLogger.files.L1GtUnpacker_warnings = cms.untracked.PSet( 
        threshold = cms.untracked.string('WARNING'),
        WARNING = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        L1GlobalTriggerRawToDigi = cms.untracked.PSet( limit = cms.untracked.int32(-1) ) 
        )

process.MessageLogger.files.L1GtUnpacker_info = cms.untracked.PSet( 
        threshold = cms.untracked.string('INFO'),
        INFO = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        WARNING = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        L1GtTrigReport = cms.untracked.PSet( limit = cms.untracked.int32(-1) ) 
        )

process.MessageLogger.files.L1GtUnpacker = cms.untracked.PSet( 
        threshold = cms.untracked.string('DEBUG'),
        DEBUG = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        INFO = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        WARNING = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        L1GlobalTriggerRawToDigi = cms.untracked.PSet( limit = cms.untracked.int32(-1) ) 
        )


# summary
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

# output 

process.outputL1GtUnpack = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('L1GtUnpacker.root'),
    # keep only unpacked data in the ROOT file
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_l1GtUnpack_*_*')
)

process.outpath = cms.EndPath(process.outputL1GtUnpack)

