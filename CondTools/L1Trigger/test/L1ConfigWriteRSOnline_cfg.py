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

# arguments for setting object keys by hand
options.register('L1MuDTTFMasksRcdKey',
                 '', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Object key")
options.register('L1MuGMTChannelMaskRcdKey',
                 '', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Object key")
options.register('L1RCTChannelMaskRcdKey',
                 '', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Object key")
options.register('L1GctChannelMaskRcdKey',
                 '', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Object key")
options.register('L1GtPrescaleFactorsAlgoTrigRcdKey',
                 '', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Object key")
options.register('L1GtPrescaleFactorsTechTrigRcdKey',
                 '', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Object key")
options.register('L1GtTriggerMaskAlgoTrigRcdKey',
                 '', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Object key")
options.register('L1GtTriggerMaskTechTrigRcdKey',
                 '', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Object key")
options.register('L1GtTriggerMaskVetoTechTrigRcdKey',
                 '', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Object key")

options.parseArguments()

# Define CondDB tags
from CondTools.L1Trigger.L1CondEnum_cfi import L1CondEnum
from CondTools.L1Trigger.L1O2OTags_cfi import initL1O2OTags
initL1O2OTags()

if options.keysFromDB == 1:
    process.load("CondTools.L1Trigger.L1ConfigRSKeys_cff")
else:
    process.load("CondTools.L1Trigger.L1TriggerKeyDummy_cff")
    from CondTools.L1Trigger.L1RSSubsystemParams_cfi import initL1RSSubsystems
    initL1RSSubsystems( tagBaseVec = initL1O2OTags.tagBaseVec,
                        L1MuDTTFMasksRcdKey = options.L1MuDTTFMasksRcdKey,
                        L1MuGMTChannelMaskRcdKey = options.L1MuGMTChannelMaskRcdKey,
                        L1RCTChannelMaskRcdKey = options.L1RCTChannelMaskRcdKey,
                        L1GctChannelMaskRcdKey = options.L1GctChannelMaskRcdKey,
                        L1GtPrescaleFactorsAlgoTrigRcdKey = options.L1GtPrescaleFactorsAlgoTrigRcdKey,
                        L1GtPrescaleFactorsTechTrigRcdKey = options.L1GtPrescaleFactorsTechTrigRcdKey,
                        L1GtTriggerMaskAlgoTrigRcdKey = options.L1GtTriggerMaskAlgoTrigRcdKey,
                        L1GtTriggerMaskTechTrigRcdKey = options.L1GtTriggerMaskTechTrigRcdKey,
                        L1GtTriggerMaskVetoTechTrigRcdKey = options.L1GtTriggerMaskVetoTechTrigRcdKey,
                        includeL1RCTNoisyChannelMask = False )
    process.L1TriggerKeyDummy.objectKeys = initL1RSSubsystems.params.recordInfo                        

# Get L1TriggerKeyList from DB
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.outputDB = cms.ESSource("PoolDBESSource",
                                process.CondDBCommon,
                                toGet = cms.VPSet(cms.PSet(
    record = cms.string('L1TriggerKeyListRcd'),
    tag = cms.string('L1TriggerKeyList_' + initL1O2OTags.tagBaseVec[ L1CondEnum.L1TriggerKeyList ] )
    )),
                                RefreshEachRun=cms.untracked.bool(True)
                                )
process.outputDB.connect = options.outputDBConnect
process.outputDB.DBParameters.authenticationPath = options.outputDBAuth

# Generate configuration data
process.load("CondTools.L1Trigger.L1ConfigRSPayloads_cff")

# writer modules
from CondTools.L1Trigger.L1CondDBPayloadWriter_cff import initPayloadWriter
initPayloadWriter( process,
                   outputDBConnect = options.outputDBConnect,
                   outputDBAuth = options.outputDBAuth,
                   tagBaseVec = initL1O2OTags.tagBaseVec )
process.L1CondDBPayloadWriter.writeL1TriggerKey = cms.bool(False)

if options.logTransactions == 1:
#    initPayloadWriter.outputDB.logconnect = cms.untracked.string('oracle://cms_orcon_prod/CMS_COND_31X_POPCONLOG')
    initPayloadWriter.outputDB.logconnect = cms.untracked.string('sqlite_file:l1o2o-log.db')
    process.L1CondDBPayloadWriter.logTransactions = True

if options.overwriteKeys == 0:
    process.L1CondDBPayloadWriter.overwriteKeys = False
else:
    process.L1CondDBPayloadWriter.overwriteKeys = True

if options.forceUpdate == 1:
    process.L1CondDBIOVWriter.forceUpdate = True

from CondTools.L1Trigger.L1CondDBIOVWriter_cff import initIOVWriter
initIOVWriter( process,
               outputDBConnect = options.outputDBConnect,
               outputDBAuth = options.outputDBAuth,
               tagBaseVec = initL1O2OTags.tagBaseVec,
               tscKey = '' )
process.L1CondDBIOVWriter.logKeys = True

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

# CORAL debugging
#process.outputDB.DBParameters.messageLevel = cms.untracked.int32(3)

process.p = cms.Path(process.L1CondDBPayloadWriter*process.L1CondDBIOVWriter)
