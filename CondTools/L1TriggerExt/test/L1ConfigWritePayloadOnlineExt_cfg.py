import FWCore.ParameterSet.Config as cms

process = cms.Process("L1ConfigWritePayloadOnline")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.enable = cms.untracked.bool(True)
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('*')
process.MessageLogger.suppressInfo = cms.untracked.vstring('L1TMuonBarrelParamsOnlineProd') # suppressDebug, suppressWarning

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
options.register('onlineDBConnect',
                 'oracle://CMS_OMDS_LB/CMS_TRG_R', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Connection string for online DB")
options.register('onlineDBAuth',
                 '.', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Authentication path for online DB")
options.register('protoDBConnect',
                 'oracle://cms_orcon_prod/CMS_CONDITIONS', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Connection string for prototypes' DB")
options.register('protoDBAuth',
                 '.', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Authentication path for prototypes' DB")
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
options.register('subsystemLabels',
                 'uGT,uGTrs,uGMT,CALO,BMTF,OMTF,EMTF', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Comma-separated list of specific payloads to be processed")
options.register('tagUpdate',
                 '', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Comma-separated list of column-separated pairs relating type to a new tagBase")
options.register('unsafe',
                 '', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Comma-separated list of systems that we do not care about anymore")
options.register('dropFromJob',
                 '', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Comma-separated list of systems for which we must not create WriterProxy")

options.parseArguments()

# Generate L1TriggerKeyExt from OMDS
process.load("CondTools.L1TriggerExt.L1SubsystemKeysOnlineExt_cfi")
process.L1SubsystemKeysOnlineExt.tscKey = cms.string( options.tscKey )
process.L1SubsystemKeysOnlineExt.rsKey  = cms.string( options.rsKey )
process.L1SubsystemKeysOnlineExt.onlineDB = cms.string( options.onlineDBConnect )
process.L1SubsystemKeysOnlineExt.onlineAuthentication = cms.string( options.onlineDBAuth )

process.load("CondTools.L1TriggerExt.L1ConfigTSCKeysExt_cff")
from CondTools.L1TriggerExt.L1ConfigTSCKeysExt_cff import setTSCKeysDB, liftKeySafetyFor
setTSCKeysDB( process, options.onlineDBConnect, options.onlineDBAuth )
liftKeySafetyFor( process, options.unsafe.split(',') )

process.load("CondTools.L1TriggerExt.L1TriggerKeyOnlineExt_cfi")
process.L1TriggerKeyOnlineExt.subsystemLabels = cms.vstring( options.subsystemLabels.split(',') )

# Generate configuration data from OMDS
process.load("CondTools.L1TriggerExt.L1ConfigTSCPayloadsExt_cff")
from CondTools.L1TriggerExt.L1ConfigTSCPayloadsExt_cff import setTSCPayloadsDB, liftPayloadSafetyFor
setTSCPayloadsDB( process, options.onlineDBConnect, options.onlineDBAuth, options.protoDBConnect, options.protoDBAuth )
liftPayloadSafetyFor( process, options.unsafe.split(',') )
print( "Lifted transaction safe for:", options.unsafe.split(',') )

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

process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = options.outputDBConnect if options.copyNonO2OPayloads == 0 else options.copyDBConnect

# Suppress warnings, not actually used, except for copyNonO2OPayloads
process.outputDB = cms.ESSource("PoolDBESSource",
                                process.CondDB,
                                toGet = cms.VPSet(cms.PSet(
    record = cms.string('L1TriggerKeyListExtRcd'),
    tag = cms.string( "L1TriggerKeyListExt_" + initL1O2OTagsExt.tagBaseVec[ L1CondEnumExt.L1TriggerKeyListExt ] )
    )),
                                RefreshEachRun=cms.untracked.bool(True)
                                )

if options.copyNonO2OPayloads == 0:
    process.outputDB.DBParameters.authenticationPath = options.outputDBAuth
    process.source = cms.Source("EmptySource")
else:
    process.outputDB.DBParameters.authenticationPath = options.copyDBAuth
    process.source = cms.Source("EmptyIOVSource",
                                timetype = cms.string('runnumber'),
                                firstValue = cms.uint64(4294967295),
                                lastValue = cms.uint64(4294967295),
                                interval = cms.uint64(1) )
                            
systems = process.L1CondDBPayloadWriterExt.sysWriters
## still need a method to handle exceptions for existing payloads in the CondDB
systems.remove("L1TMuonEndCapForestO2ORcd@L1TMuonEndCapForest")
systems.remove("L1TMuonOverlapParamsO2ORcd@L1TMuonOverlapParams")
if "uGMT" in options.dropFromJob:
    systems.remove("L1TMuonGlobalParamsO2ORcd@L1TMuonGlobalParams")
if "EMTF" in options.dropFromJob:
    systems.remove("L1TMuonEndCapParamsO2ORcd@L1TMuonEndCapParams")
    ## still need a method to handle exceptions for existing payloads in the CondDB
if "OMTF" in options.dropFromJob:
    systems.remove("L1TMuonOverlapFwVersionO2ORcd@L1TMuonOverlapFwVersion")
    ## still need a method to handle exceptions for existing payloads in the CondDB
if "BMTF" in options.dropFromJob:
    systems.remove("L1TMuonBarrelParamsO2ORcd@L1TMuonBarrelParams")
if "CALO" in options.dropFromJob:
    systems.remove("L1TCaloParamsO2ORcd@CaloParams")
if "uGT" in options.dropFromJob:
    systems.remove("L1TUtmTriggerMenuO2ORcd@L1TUtmTriggerMenu")
if "uGTrs" in options.dropFromJob:
    systems.remove("L1TGlobalPrescalesVetosFractO2ORcd@L1TGlobalPrescalesVetosFract")
print( "Will create only the following writers:", process.L1CondDBPayloadWriterExt.sysWriters )

process.p = cms.Path(process.L1CondDBPayloadWriterExt)
