import FWCore.ParameterSet.Config as cms

process = cms.Process("L1ConfigWriteIOVOnline")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.enable = cms.untracked.bool(True)
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

# Get L1TriggerKeyList from DB
process.load("CondCore.DBCommon.CondDBCommon_cfi")

import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing()
options.register('tscKey',
                 '', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "TSC key")
options.register('runNumber',
                 0, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Run number")
options.register('outputDBConnect',
                 'sqlite_file:l1config.db', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Connection string for output DB")
options.register('outputDBAuth',
                 '.', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Authentication path for output DB")
options.register('logTransactions',
                 1, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Record transactions in log DB")
options.parseArguments()

# Define CondDB tags
from CondTools.L1Trigger.L1CondEnum_cfi import L1CondEnum
from CondTools.L1Trigger.L1O2OTags_cfi import initL1O2OTags
initL1O2OTags()

# writer modules
from CondTools.L1Trigger.L1CondDBIOVWriter_cff import initIOVWriter
initIOVWriter( process,
               outputDBConnect = options.outputDBConnect,
               outputDBAuth = options.outputDBAuth,
               tagBaseVec = initL1O2OTags.tagBaseVec,
               tscKey = options.tscKey )

if options.logTransactions == 1:
#    initIOVWriter.outputDB.logconnect = cms.untracked.string('oracle://cms_orcon_prod/CMS_COND_31X_POPCONLOG')
    initIOVWriter.outputDB.logconnect = cms.untracked.string('sqlite_file:l1o2o-log.db')
    process.L1CondDBIOVWriter.logTransactions = True

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(options.runNumber),
    lastValue = cms.uint64(options.runNumber),
    interval = cms.uint64(1)
)

process.outputDB = cms.ESSource("PoolDBESSource",
                                process.CondDBCommon,
                                toGet = cms.VPSet(cms.PSet(
    record = cms.string('L1TriggerKeyListRcd'),
    tag = cms.string('L1TriggerKeyList_' + initL1O2OTags.tagBaseVec[ L1CondEnum.L1TriggerKeyList ])
    )),
                                RefreshEachRun=cms.untracked.bool(True)
                                )
process.outputDB.connect = cms.string(options.outputDBConnect)
process.outputDB.DBParameters.authenticationPath = options.outputDBAuth

# CORAL debugging
process.outputDB.DBParameters.messageLevel = cms.untracked.int32(3)

process.p = cms.Path(process.L1CondDBIOVWriter)
