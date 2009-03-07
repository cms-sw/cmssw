import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("L1ConfigWritePayloadOnline")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.placeholder = cms.untracked.bool(False)
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

process.load("CondCore.DBCommon.CondDBCommon_cfi")

options = VarParsing.VarParsing()
options.register('tscKey',
                 '', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "TSC key")
options.register('tagBase',
                 'CRAFT', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "IOV tags = object_{tagBase}_hlt")
options.register('orconConnect',
                 'sqlite_file:l1config.db', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Connection string for ORCON")
options.register('orconAuth',
                 '.', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Authentication path for ORCON")
options.parseArguments()

# Generate L1TriggerKey and configuration data from OMDS
process.load("CondTools.L1Trigger.L1SubsystemKeysOnline_cfi")
process.L1SubsystemKeysOnline.tscKey = cms.string( options.tscKey )

process.load("L1TriggerConfig.CSCTFConfigProducers.CSCTFObjectKeysOnline_cfi")
process.load("L1TriggerConfig.RPCTriggerConfig.L1RPCObjectKeysOnline_cfi")
process.load("L1TriggerConfig.GMTConfigProducers.L1MuGMTParametersKeysOnlineProd_cfi")
process.load("L1TriggerConfig/L1ScalesProducers.L1MuTriggerScaleKeysOnlineProd_cfi")
process.L1MuTriggerScaleKeysOnlineProd.subsystemLabel = 'GMTScales'
process.load("L1TriggerConfig.RCTConfigProducers.L1RCTObjectKeysOnline_cfi")
process.load("L1TriggerConfig.L1GtConfigProducers.l1GtTscObjectKeysOnline_cfi")
process.l1GtTscObjectKeysOnline.EnableL1GtTriggerMenu = False
process.l1GtTscObjectKeysOnline.EnableL1GtPsbSetup = False

process.load("CondTools.L1Trigger.L1TriggerKeyOnline_cfi")
process.L1TriggerKeyOnline.subsystemLabels = cms.vstring( 'CSCTF',
                                                          'RPC',
                                                          'GMT',
                                                          'GMTScales',
                                                          'RCT',
                                                          'GT' )

process.load("L1TriggerConfig.CSCTFConfigProducers.CSCTFConfigOnline_cfi")
process.load("L1TriggerConfig.RPCTriggerConfig.L1RPCConfigOnline_cfi")
process.load("L1TriggerConfig.GMTConfigProducers.L1MuGMTParametersOnlineProducer_cfi")
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerPtScaleOnlineProducer_cfi")
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerScalesOnlineProducer_cfi")
process.L1MuGMTParametersOnlineProducer.ignoreVersionMismatch = True
process.load("L1TriggerConfig.RCTConfigProducers.L1RCTParametersOnline_cfi")
process.load("L1TriggerConfig.L1GtConfigProducers.l1GtParametersOnline_cfi")

# writer modules
process.load("CondTools.L1Trigger.L1CondDBPayloadWriter_cfi")
process.L1CondDBPayloadWriter.offlineDB = cms.string(options.orconConnect)
process.L1CondDBPayloadWriter.offlineAuthentication = options.orconAuth
process.L1CondDBPayloadWriter.L1TriggerKeyListTag = cms.string( 'L1TriggerKeyList_' + options.tagBase + '_hlt')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(4294967295),
    lastValue = cms.uint64(4294967295),
    interval = cms.uint64(1)
)

process.orcon = cms.ESSource("PoolDBESSource",
    process.CondDBCommon,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('L1TriggerKeyListRcd'),
        tag = cms.string( "L1TriggerKeyList_" + options.tagBase + "_hlt" )
    ))
)
process.orcon.connect = cms.string(options.orconConnect)
process.orcon.DBParameters.authenticationPath = options.orconAuth

process.p = cms.Path(process.L1CondDBPayloadWriter)


