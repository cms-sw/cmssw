import FWCore.ParameterSet.Config as cms

process = cms.Process("L1ConfigWriteIOVOnlineExt")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.enable = cms.untracked.bool(True)
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('*')


import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing()
options.register('tscKey',
                 '', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "TSC key")
options.register('rsKey',
                 '', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "RS key")
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
options.register('forceUpdate',
                 0, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Check all record IOVs even if L1TriggerKey unchanged")
options.register('logTransactions',
                 1, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Record transactions in log DB")
options.register('tagUpdate',
                 '', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Comma-separated list of column-separated pairs relating type to a new tagBase")
options.parseArguments()

# Define CondDB tags
from CondTools.L1TriggerExt.L1CondEnumExt_cfi import L1CondEnumExt
from CondTools.L1TriggerExt.L1O2OTagsExt_cfi import initL1O2OTagsExt
initL1O2OTagsExt()

# Override the tag bases if instructed to do so
if options.tagUpdate :
    for type2tagBase in options.tagUpdate.split(',') :
        (t,tagBase) = type2tagBase.split(':')
        index = L1CondEnumExt.__dict__[t]
        initL1O2OTagsExt.tagBaseVec[index] = tagBase

# writer modules
from CondTools.L1TriggerExt.L1CondDBIOVWriterExt_cff import initIOVWriterExt
initIOVWriterExt( process,
               outputDBConnect = options.outputDBConnect,
               outputDBAuth = options.outputDBAuth,
               tagBaseVec = initL1O2OTagsExt.tagBaseVec,
               tscKey = options.tscKey,
               rsKey  = options.rsKey )

if options.forceUpdate == 1:
    process.L1CondDBIOVWriterExt.forceUpdate = True

if options.logTransactions == 1:
#    initIOVWriterExt.outputDB.logconnect = cms.untracked.string('oracle://cms_orcon_prod/CMS_COND_31X_POPCONLOG')
    initIOVWriterExt.outputDB.logconnect = cms.untracked.string('sqlite_file:l1o2o-log.db')
    process.L1CondDBIOVWriterExt.logTransactions = True

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(options.runNumber),
    lastValue = cms.uint64(options.runNumber),
    interval = cms.uint64(1)
)

# Get L1TriggerKeyListExtExt from DB
process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = cms.string(options.outputDBConnect)

process.outputDB = cms.ESSource("PoolDBESSource",
                                process.CondDB,
                                toGet = cms.VPSet(cms.PSet(
    record = cms.string('L1TriggerKeyListExtRcd'),
    tag = cms.string('L1TriggerKeyListExt_' + initL1O2OTagsExt.tagBaseVec[ L1CondEnumExt.L1TriggerKeyListExt ])
    )),
                                RefreshEachRun=cms.untracked.bool(True)
                                )

process.outputDB.DBParameters.authenticationPath = options.outputDBAuth

# CORAL debugging
process.outputDB.DBParameters.messageLevel = cms.untracked.int32(3)

process.p = cms.Path(process.L1CondDBIOVWriterExt)
