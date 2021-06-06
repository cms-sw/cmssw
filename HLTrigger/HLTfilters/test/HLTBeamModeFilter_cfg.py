from __future__ import print_function
#
# cfg file to test HLTBeamModeFilter
# it requires as input:
#   a RAW data file (with FED 812 included) or 
#   a digi data file, with L1GlobalTriggerEvmReadoutRecord product valid
#
# the input sample is chosen in L1Trigger.GlobalTriggerAnalyzer.UserOptions_cff
#
# V.M. Ghete 2010-05-31
#

import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process('TestHLTBeamModeFilter')

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

process.load('Configuration.StandardSequences.Geometry_cff')

# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.GlobalTag.globaltag = useGlobalTag+'::All'





# this module
process.load('HLTrigger.HLTfilters.hltBeamModeFilter_cfi')
 
# replacing arguments for hltBeamModeFilter

    # InputTag for the L1 Global Trigger EVM readout record
    #   gtDigis        GT Emulator
    #   l1GtEvmUnpack  GT EVM Unpacker (default module name) 
    #   gtEvmDigis     GT EVM Unpacker in RawToDigi standard sequence  
    #
    #   cloned GT unpacker in HLT = gtEvmDigis
process.hltBeamModeFilter.L1GtEvmReadoutRecordTag = cms.InputTag("gtEvmDigis")

# vector of allowed beam modes (see enumeration in header file for implemented values)
# default value: 11 (STABLE)
#process.hltBeamModeFilter.AllowedBeamMode = cms.vuint32(11)
process.hltBeamModeFilter.AllowedBeamMode = cms.vuint32(9, 10, 11)


# path to be run
# for RAW data, run first the RAWTODIGI 
if (dataType == 'RAW') and not (useRelValSample) :
    process.load('Configuration/StandardSequences/RawToDigi_Data_cff')
    process.gtEvmDigis.UnpackBxInEvent = cms.int32(1)
    # set EVM unpacker to verbose
    process.gtEvmDigis.Verbosity = cms.untracked.int32(1)

    process.p = cms.Path(process.RawToDigi * process.hltBeamModeFilter)

elif (dataType == 'RAW') and (useRelValSample) :
    process.load('Configuration/StandardSequences/RawToDigi_cff')
    process.gtEvmDigis.UnpackBxInEvent = cms.int32(1)
    # set EVM unpacker to verbose
    process.gtEvmDigis.Verbosity = cms.untracked.int32(1)

    process.p = cms.Path(process.RawToDigi * process.hltBeamModeFilter)
    
else :        
    # it does not run on standard RECO, but I let the path here (exit in the filter) 
    process.p = cms.Path(process.hltBeamModeFilter)

# services

# Message Logger
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.debugModules = ['gtEvmDigis', 'hltBeamModeFilter']
process.MessageLogger.L1GlobalTriggerEvmRawToDigi=dict()
process.MessageLogger.HLTBeamModeFilter=dict()

process.MessageLogger.cerr.threshold = 'DEBUG'
#process.MessageLogger.cerr.threshold = 'INFO'
#process.MessageLogger.cerr.threshold = 'WARNING'
#process.MessageLogger.cerr.threshold = 'ERROR'

process.MessageLogger.cerr.DEBUG = cms.untracked.PSet( limit = cms.untracked.int32(0) )
process.MessageLogger.cerr.INFO = cms.untracked.PSet( limit = cms.untracked.int32(0) )
process.MessageLogger.cerr.WARNING = cms.untracked.PSet( limit = cms.untracked.int32(0) )
process.MessageLogger.cerr.ERROR = cms.untracked.PSet( limit = cms.untracked.int32(0) )

process.MessageLogger.cerr.L1GlobalTriggerEvmRawToDigi = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
process.MessageLogger.cerr.HLTBeamModeFilter = cms.untracked.PSet( limit = cms.untracked.int32(-1) )


