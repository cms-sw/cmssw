from __future__ import print_function
#
# cfg file to unpack RAW L1 GT EVM data
 
# V M Ghete 2009-04-03

import FWCore.ParameterSet.Config as cms

# process
process = cms.Process('TestL1GtEvmUnpacker')

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

process.load('Configuration.StandardSequences.Geometry_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.GlobalTag.globaltag = useGlobalTag

# L1 GT/GMT EvmUnpack
process.load("EventFilter.L1GlobalTriggerRawToDigi.l1GtEvmUnpack_cfi")

# input tag for GT readout collection (before CMSSW_5_0_X)
#     source        = hardware record
#
#if useRelValSample == True :
#    evmGtInputTag = 'rawDataCollector'
#else :
#    evmGtInputTag = 'rawDataCollector'

evmGtInputTag = 'rawDataCollector'

process.l1GtEvmUnpack.EvmGtInputTag = evmGtInputTag

# Active Boards Mask

# no board masked (default)
#process.l1GtEvmUnpack.ActiveBoardsMask = 0xFFFF
    
# GTFE only in the record
#process.l1GtEvmUnpack.ActiveBoardsMask = 0x0000
     

# BxInEvent to be EvmUnpacked
# all available BxInEvent (default)
#process.l1GtEvmUnpack.UnpackBxInEvent = -1 

# BxInEvent = 0 (L1A)
#process.l1GtEvmUnpack.UnpackBxInEvent = 1 

# 3 BxInEvent (F, 0, 1)  
#process.l1GtEvmUnpack.UnpackBxInEvent = 3 

# length of BST message (in bytes)
# if negative, take it from event setup
#process.l1GtEvmUnpack.BstLengthBytes = 52

# set it to verbose
process.l1GtEvmUnpack.Verbosity = cms.untracked.int32(1)


# path to be run
process.p = cms.Path(process.l1GtEvmUnpack)

# Message Logger
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.debugModules = ['l1GtEvmUnpack', 'l1GtTrigReport']
process.MessageLogger.categories.append('L1GlobalTriggerEvmRawToDigi')
process.MessageLogger.categories.append('L1GtTrigReport')
process.MessageLogger.destinations = ['L1GtEvmUnpacker_errors', 
                                      'L1GtEvmUnpacker_warnings', 
                                      'L1GtEvmUnpacker_info', 
                                      'L1GtEvmUnpacker'
                                      ]
process.MessageLogger.statistics = []

process.MessageLogger.L1GtEvmUnpacker_errors = cms.untracked.PSet( 
        threshold = cms.untracked.string('ERROR'),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
        L1GlobalTriggerEvmRawToDigi = cms.untracked.PSet( limit = cms.untracked.int32(-1) ) 
       )

process.MessageLogger.L1GtEvmUnpacker_warnings = cms.untracked.PSet( 
        threshold = cms.untracked.string('WARNING'),
        WARNING = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        L1GlobalTriggerEvmRawToDigi = cms.untracked.PSet( limit = cms.untracked.int32(-1) ) 
        )

process.MessageLogger.L1GtEvmUnpacker_info = cms.untracked.PSet( 
        threshold = cms.untracked.string('INFO'),
        INFO = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        WARNING = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        L1GtTrigReport = cms.untracked.PSet( limit = cms.untracked.int32(-1) ) 
        )

process.MessageLogger.L1GtEvmUnpacker = cms.untracked.PSet( 
        threshold = cms.untracked.string('DEBUG'),
        DEBUG = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        INFO = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        WARNING = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        L1GlobalTriggerEvmRawToDigi = cms.untracked.PSet( limit = cms.untracked.int32(-1) ) 
        )

# summary
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

# output 

process.outputL1GtEvmUnpack = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('L1GtEvmUnpacker.root'),
    # keep only unpacked data in the ROOT file
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_l1GtEvmUnpack_*_*')
)

process.outpath = cms.EndPath(process.outputL1GtEvmUnpack)
