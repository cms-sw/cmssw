import FWCore.ParameterSet.Config as cms

process = cms.Process("L1ConfigWritePayloadOnline")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.placeholder = cms.untracked.bool(False)
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

process.load("CondCore.DBCommon.CondDBCommon_cfi")

import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing()
options.register('tscKey',
                 '', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "TSC key")
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
options.register('overwriteKeys',
                 0, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Overwrite existing keys")
options.register('logTransactions',
                 1, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Record transactions in log DB")
options.register('copyNonO2OPayloads',
                 0, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Copy DTTF TSC payloads from ORCON")
options.register('copyDBConnect',
                 'sqlite_file:l1config.db', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Connection string for copy DB")
options.register('copyDBAuth',
                 '.', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Authentication path for copy DB")
options.parseArguments()

# Generate L1TriggerKeyExt from OMDS
process.load("CondTools.L1TriggerExt.L1SubsystemKeysOnlineExt_cfi")
process.L1SubsystemKeysOnlineExt.tscKey = cms.string( options.tscKey )
process.load("CondTools.L1TriggerExt.L1ConfigTSCKeysExt_cff")
process.load("CondTools.L1TriggerExt.L1TriggerKeyOnlineExt_cfi")
process.L1TriggerKeyOnlineExt.subsystemLabels = cms.vstring(
                                                          'uGT'
                                                        )

# Generate configuration data from OMDS
process.load("CondTools.L1TriggerExt.L1ConfigTSCPayloadsExt_cff")

# Define CondDB tags
from CondTools.L1TriggerExt.L1CondEnumExt_cfi import L1CondEnumExt
from CondTools.L1TriggerExt.L1O2OTagsExt_cfi import initL1O2OTagsExt
initL1O2OTagsExt()

# writer modules
from CondTools.L1TriggerExt.L1CondDBPayloadWriterExt_cff import initPayloadWriterExt
initPayloadWriterExt( process,
                   outputDBConnect = options.outputDBConnect,
                   outputDBAuth = options.outputDBAuth,
                   tagBaseVec = initL1O2OTagsExt.tagBaseVec )

if options.logTransactions == 1:
#    initPayloadWriterExt.outputDB.logconnect = cms.untracked.string('oracle://cms_orcon_prod/CMS_COND_31X_POPCONLOG')
    initPayloadWriterExt.outputDB.logconnect = cms.untracked.string('sqlite_file:l1o2o-log.db')
    process.L1CondDBPayloadWriterExt.logTransactions = True

if options.overwriteKeys == 0:
    process.L1CondDBPayloadWriterExt.overwriteKeys = False
else:
    process.L1CondDBPayloadWriterExt.overwriteKeys = True
                
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

# Suppress warnings, not actually used, except for copyNonO2OPayloads
process.outputDB = cms.ESSource("PoolDBESSource",
                                process.CondDBCommon,
                                toGet = cms.VPSet(cms.PSet(
    record = cms.string('L1TriggerKeyListExtRcd'),
    tag = cms.string( "L1TriggerKeyListExt_" + initL1O2OTagsExt.tagBaseVec[ L1CondEnumExt.L1TriggerKeyListExt ] )
    )),
                                RefreshEachRun=cms.untracked.bool(True)
                                )

if options.copyNonO2OPayloads == 0:
    process.outputDB.connect = options.outputDBConnect
    process.outputDB.DBParameters.authenticationPath = options.outputDBAuth
    process.source = cms.Source("EmptySource")
else:
    process.outputDB.connect = options.copyDBConnect
    process.outputDB.DBParameters.authenticationPath = options.copyDBAuth
    process.source = cms.Source("EmptyIOVSource",
                                timetype = cms.string('runnumber'),
                                firstValue = cms.uint64(4294967295),
                                lastValue = cms.uint64(4294967295),
                                interval = cms.uint64(1) )
                            
process.p = cms.Path(process.L1CondDBPayloadWriterExt)
