import FWCore.ParameterSet.Config as cms

process = cms.Process("L1ConfigWriteRSPayloadOnline")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.enable = cms.untracked.bool(True)
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing()
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
options.register('logTransactions',
                 1, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Record transactions in log DB")

# arguments for setting object keys by hand
options.register('runNumber',
                 0, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Dummy argument")

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
    initL1RSSubsystemsExt( tagBaseVec = initL1O2OTagsExt.tagBaseVec )
    process.L1TriggerKeyDummyExt.objectKeys = initL1RSSubsystemsExt.params.recordInfo

# Get L1TriggerKeyListExt from DB
process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = options.outputDBConnect

process.outputDB = cms.ESSource("PoolDBESSource",
    process.CondDB,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('L1TriggerKeyListExtRcd'),
        tag = cms.string('L1TriggerKeyListExt_' + initL1O2OTagsExt.tagBaseVec[ L1CondEnumExt.L1TriggerKeyListExt ] )
    ))
)

#process.es_prefer_outputDB = cms.ESPrefer("PoolDBESSource","outputDB")
process.outputDB.DBParameters.authenticationPath = options.outputDBAuth

# Generate configuration data
process.load("CondTools.L1TriggerExt.L1ConfigRSPayloadsExt_cff")

# writer modules
from CondTools.L1TriggerExt.L1CondDBPayloadWriterExt_cff import initPayloadWriterExt
initPayloadWriterExt( process,
                   outputDBConnect = options.outputDBConnect,
                   outputDBAuth = options.outputDBAuth,
                   tagBaseVec = initL1O2OTagsExt.tagBaseVec,
process.L1CondDBPayloadWriterExt.writeL1TriggerKey = cms.bool(False)

if options.logTransactions == 1:
    initPayloadWriterExt.outputDB.logconnect = cms.untracked.string('oracle://cms_orcon_prod/CMS_COND_31X_POPCONLOG')
    process.L1CondDBPayloadWriterExt.logTransactions = True

if options.overwriteKeys == 0:
    process.L1CondDBPayloadWriterExt.overwriteKeys = False
else:
    process.L1CondDBPayloadWriterExt.overwriteKeys = True
                
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.p = cms.Path(process.L1CondDBPayloadWriterExt)
