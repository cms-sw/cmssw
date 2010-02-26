import FWCore.ParameterSet.Config as cms

process = cms.Process("L1ConfigWriteRSIOVOnline")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.placeholder = cms.untracked.bool(False)
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing()
options.register('runNumber',
                 0, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Run number")
options.register('tagBase',
                 'CRAFT_hlt', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "IOV tags = object_{tagBase}")
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

# arguments for setting object keys by hand
options.register('L1MuDTTFMasksRcdKey',
                 'dummy', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Object key")
options.register('L1MuGMTChannelMaskRcdKey',
                 'dummy', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Object key")
options.register('L1RCTChannelMaskRcdKey',
                 'dummy', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Object key")
options.register('L1GctChannelMaskRcdKey',
                 'dummy', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Object key")
options.register('L1GtPrescaleFactorsAlgoTrigRcdKey',
                 'dummy', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Object key")
options.register('L1GtPrescaleFactorsTechTrigRcdKey',
                 'dummy', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Object key")
options.register('L1GtTriggerMaskAlgoTrigRcdKey',
                 'dummy', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Object key")
options.register('L1GtTriggerMaskTechTrigRcdKey',
                 'dummy', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Object key")
options.register('L1GtTriggerMaskVetoTechTrigRcdKey',
                 'dummy', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Object key")

options.parseArguments()

if options.keysFromDB == 1:
    process.load("CondTools.L1Trigger.L1ConfigRSKeys_cff")
else:
    process.load("CondTools.L1Trigger.L1TriggerKeyDummy_cff")
    from CondTools.L1Trigger.L1RSSubsystemParams_cfi import initL1RSSubsystems
    initL1RSSubsystems( tagBase = options.tagBase,
                        L1MuDTTFMasksRcdKey = options.L1MuDTTFMasksRcdKey,
                        L1MuGMTChannelMaskRcdKey = options.L1MuGMTChannelMaskRcdKey,
                        L1RCTChannelMaskRcdKey = options.L1RCTChannelMaskRcdKey,
                        L1GctChannelMaskRcdKey = options.L1GctChannelMaskRcdKey,
                        L1GtPrescaleFactorsAlgoTrigRcdKey = options.L1GtPrescaleFactorsAlgoTrigRcdKey,
                        L1GtPrescaleFactorsTechTrigRcdKey = options.L1GtPrescaleFactorsTechTrigRcdKey,
                        L1GtTriggerMaskAlgoTrigRcdKey = options.L1GtTriggerMaskAlgoTrigRcdKey,
                        L1GtTriggerMaskTechTrigRcdKey = options.L1GtTriggerMaskTechTrigRcdKey,
                        L1GtTriggerMaskVetoTechTrigRcdKey = options.L1GtTriggerMaskVetoTechTrigRcdKey )
    process.L1TriggerKeyDummy.objectKeys = initL1RSSubsystems.params.recordInfo

# Get L1TriggerKeyList from DB
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.outputDB = cms.ESSource("PoolDBESSource",
                                process.CondDBCommon,
                                toGet = cms.VPSet(cms.PSet(
    record = cms.string('L1TriggerKeyListRcd'),
    tag = cms.string('L1TriggerKeyList_' + options.tagBase )
    ))
                                )
process.outputDB.connect = options.outputDBConnect
process.outputDB.DBParameters.authenticationPath = options.outputDBAuth

# writer modules
from CondTools.L1Trigger.L1CondDBIOVWriter_cff import initIOVWriter
initIOVWriter( process,
               outputDBConnect = options.outputDBConnect,
               outputDBAuth = options.outputDBAuth,
               tagBase = options.tagBase,
               tscKey = '' )
process.L1CondDBIOVWriter.logKeys = True
initIOVWriter.outputDB.logconnect = cms.untracked.string('sqlite_file:o2o_iov_log.db')
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

process.p = cms.Path(process.L1CondDBIOVWriter)
