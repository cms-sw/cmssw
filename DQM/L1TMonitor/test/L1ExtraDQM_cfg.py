#
# cfg file to run L1ExtraDQM, in conjunction with a standard 
# DQM/Integration/l1t_sourceclient-live_cfg.py 
#
# V M Ghete 2010-02-24

import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("DQM")

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

#
# load and configure modules via Global Tag
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions

process.load('Configuration.StandardSequences.Geometry_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.GlobalTag.globaltag = useGlobalTag+'::All'


#  DQM SERVICES
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

#  DQM SOURCES
process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.load("DQM/L1TMonitor/L1TMonitor_cff")

process.DQM.collectorHost = "srv-c2d05-12"
process.DQM.collectorPort = 9190

# Message Logger
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.debugModules = ['l1ExtraDQM']
process.MessageLogger.categories.append('L1ExtraDQM')
process.MessageLogger.categories.append('L1RetrieveL1Extra')
process.MessageLogger.categories.append('L1GetHistLimits')

process.MessageLogger.cerr.default.limit = 0
process.MessageLogger.cerr.FwkJob.limit = 0
process.MessageLogger.cerr.FwkReport.limit = 0
process.MessageLogger.cerr.FwkSummary.limit = 0

process.MessageLogger.debugs = cms.untracked.PSet( 
        threshold = cms.untracked.string('DEBUG'),
        DEBUG = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        INFO = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        WARNING = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        L1ExtraDQM = cms.untracked.PSet( limit = cms.untracked.int32(-1) ), 
        L1RetrieveL1Extra = cms.untracked.PSet( limit = cms.untracked.int32(-1) ), 
        L1GetHistLimits = cms.untracked.PSet( limit = cms.untracked.int32(-1) ) 
        )

process.MessageLogger.warnings = cms.untracked.PSet( 
        threshold = cms.untracked.string('WARNING'),
        WARNING = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        L1ExtraDQM = cms.untracked.PSet( limit = cms.untracked.int32(-1) ), 
        L1RetrieveL1Extra = cms.untracked.PSet( limit = cms.untracked.int32(-1) ), 
        L1GetHistLimits = cms.untracked.PSet( limit = cms.untracked.int32(-1) ) 
        )

process.MessageLogger.errors = cms.untracked.PSet( 
        threshold = cms.untracked.string('ERROR'),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
        L1ExtraDQM = cms.untracked.PSet( limit = cms.untracked.int32(-1) ), 
        L1RetrieveL1Extra = cms.untracked.PSet( limit = cms.untracked.int32(-1) ), 
        L1GetHistLimits = cms.untracked.PSet( limit = cms.untracked.int32(-1) ) 
        )

process.p = cms.EndPath(process.dqmSaver)
process.DQMStore.verbose = 0
process.dqmSaver.convention = 'Online'
process.dqmSaver.dirName = '.'
process.dqmSaver.producer = 'DQM'
process.dqmEnv.subSystemFolder = 'L1T'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True

