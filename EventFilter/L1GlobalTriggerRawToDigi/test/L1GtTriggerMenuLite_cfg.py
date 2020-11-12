from __future__ import print_function
#
# cfg file to run the L1GtTriggerMenuLite producer  
# with the options set in UserOptions_cff.py
#
#

import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("L1T")

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
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions

process.load('Configuration.StandardSequences.Geometry_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.GlobalTag.globaltag = useGlobalTag+'::All'


# processes to be run

process.load("EventFilter.L1GlobalTriggerRawToDigi.l1GtTriggerMenuLite_cfi")

# for RAW data, run first the RAWTODIGI 
if dataType == 'RAW' :
    process.load('Configuration/StandardSequences/RawToDigi_Data_cff')
    process.p = cms.Path(process.RawToDigi+process.l1GtTriggerMenuLite)
    
else :        
    # path to be run for RECO
    process.p = cms.Path(process.l1GtTriggerMenuLite)


# Message Logger
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.debugModules = ['l1GtTriggerMenuLite']
process.MessageLogger.categories.append('L1GtTriggerMenuLiteProducer')

process.MessageLogger.cerr.default.limit = 0
process.MessageLogger.cerr.FwkReport.limit = 0
process.MessageLogger.cerr.FwkSummary.limit = 0

process.MessageLogger.destinations.extend('debugs', 'warnings', 'errors')
process.MessageLogger.debugs = cms.untracked.PSet( 
        threshold = cms.untracked.string('DEBUG'),
        DEBUG = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        INFO = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        WARNING = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        L1GtTriggerMenuLiteProducer = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
        )

process.MessageLogger.warnings = cms.untracked.PSet( 
        threshold = cms.untracked.string('WARNING'),
        WARNING = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        L1GtTriggerMenuLiteProducer = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
        )

process.MessageLogger.errors = cms.untracked.PSet( 
        threshold = cms.untracked.string('ERROR'),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
        L1GtTriggerMenuLiteProducer = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
        )

# output 
process.outputL1GtTriggerMenu = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('L1GtTriggerMenuLite_output.root'),
    # keep only unpacked data in the ROOT file
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_l1GtTriggerMenuLite_*_*')
)

process.outpath = cms.EndPath(process.outputL1GtTriggerMenu)
