# This script doesn't work yet.  PoolDBESSource does not see the IOV updates made earlier in the
# same event.

import FWCore.ParameterSet.Config as cms

process = cms.Process("L1ConfigWriteRSOnline")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.enable = cms.untracked.bool(True)
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing()
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
                 "Authentication path for outputDB")
options.register('keysFromDB',
                 1, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "1 = read keys from OMDS, 0 = read keys from command line")
options.register('overwriteKeys',
                 0, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Overwrite existing keys")
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

options.parseArguments()

# Define CondDB tags
from CondTools.L1TriggerExt.L1CondEnumExt_cfi import L1CondEnumExt
from CondTools.L1TriggerExt.L1O2OTagsExt_cfi import initL1O2OTagsExt
initL1O2OTagsExt()

if options.keysFromDB == 1:
    process.load("CondTools.L1TriggerExt.L1ConfigRSKeysExt_cff")
else:
    process.load("CondTools.L1TriggerExt.L1TriggerKeyDummyExt_cff")
    from CondTools.L1TriggerExt.L1RSSubsystemParamsExt_cfi import initL1RSSubsystemsExt
    initL1RSSubsystemsExt( tagBaseVec = initL1O2OTagsExt.tagBaseVec)
    process.L1TriggerKeyDummyExt.objectKeys = initL1RSSubsystemsExt.params.recordInfo                        

# Get L1TriggerKeyListExt from DB
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.outputDB = cms.ESSource("PoolDBESSource",
                                process.CondDBCommon,
                                toGet = cms.VPSet(cms.PSet(
    record = cms.string('L1TriggerKeyListExtRcd'),
    tag = cms.string('L1TriggerKeyListExt_' + initL1O2OTagsExt.tagBaseVec[ L1CondEnumExt.L1TriggerKeyListExt ] )
    )),
                                RefreshEachRun=cms.untracked.bool(True)
                                )
process.outputDB.connect = options.outputDBConnect
process.outputDB.DBParameters.authenticationPath = options.outputDBAuth

# Generate configuration data
process.load("CondTools.L1TriggerExt.L1ConfigRSPayloadsExt_cff")

# writer modules
from CondTools.L1TriggerExt.L1CondDBPayloadWriterExt_cff import initPayloadWriterExt
initPayloadWriterExt( process,
                   outputDBConnect = options.outputDBConnect,
                   outputDBAuth = options.outputDBAuth,
                   tagBaseVec = initL1O2OTagsExt.tagBaseVec )
process.L1CondDBPayloadWriterExt.writeL1TriggerKey = cms.bool(False)

if options.logTransactions == 1:
#    initPayloadWriterExt.outputDB.logconnect = cms.untracked.string('oracle://cms_orcon_prod/CMS_COND_31X_POPCONLOG')
    initPayloadWriterExt.outputDB.logconnect = cms.untracked.string('sqlite_file:l1o2o-log.db')
    process.L1CondDBPayloadWriterExt.logTransactions = True

if options.overwriteKeys == 0:
    process.L1CondDBPayloadWriterExt.overwriteKeys = False
else:
    process.L1CondDBPayloadWriterExt.overwriteKeys = True

from CondTools.L1TriggerExt.L1CondDBIOVWriterExt_cff import initIOVWriterExt
initIOVWriterExt( process,
               outputDBConnect = options.outputDBConnect,
               outputDBAuth = options.outputDBAuth,
               tagBaseVec = initL1O2OTagsExt.tagBaseVec,
               tscKey = '' )
process.L1CondDBIOVWriterExt.logKeys = True

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

# CORAL debugging
#process.outputDB.DBParameters.messageLevel = cms.untracked.int32(3)

process.p = cms.Path(process.L1CondDBPayloadWriterExt*process.L1CondDBIOVWriterExt)
