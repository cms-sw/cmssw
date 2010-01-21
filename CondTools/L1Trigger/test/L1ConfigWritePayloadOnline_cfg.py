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
options.parseArguments()

# Generate L1TriggerKey from OMDS
process.load("CondTools.L1Trigger.L1SubsystemKeysOnline_cfi")
process.L1SubsystemKeysOnline.tscKey = cms.string( options.tscKey )

process.load("L1TriggerConfig.CSCTFConfigProducers.CSCTFObjectKeysOnline_cfi")
process.load("L1TriggerConfig.DTTrackFinder.L1DTTFTSCObjectKeysOnline_cfi")
process.load("L1TriggerConfig.RPCTriggerConfig.L1RPCObjectKeysOnline_cfi")
process.load("L1TriggerConfig.GMTConfigProducers.L1MuGMTParametersKeysOnlineProd_cfi")
process.load("L1TriggerConfig/L1ScalesProducers.L1MuTriggerScaleKeysOnlineProd_cfi")
process.L1MuTriggerScaleKeysOnlineProd.subsystemLabel = 'GMTScales'
process.load("L1TriggerConfig.RCTConfigProducers.L1RCTObjectKeysOnline_cfi")
process.load("L1TriggerConfig.GctConfigProducers.L1GctTSCObjectKeysOnline_cfi")
process.load("L1TriggerConfig.L1GtConfigProducers.l1GtTscObjectKeysOnline_cfi")
#process.l1GtTscObjectKeysOnline.EnableL1GtTriggerMenu = False
#process.l1GtTscObjectKeysOnline.EnableL1GtPsbSetup = False

process.load("CondTools.L1Trigger.L1TriggerKeyOnline_cfi")
process.L1TriggerKeyOnline.subsystemLabels = cms.vstring( 'CSCTF',
                                                          'DTTF',
                                                          'RPC',
                                                          'GMT',
                                                          'GMTScales',
                                                          'RCT',
                                                          'GCT',
                                                          'GT' )

# Generate configuration data from OMDS
process.load("L1TriggerConfig.CSCTFConfigProducers.CSCTFConfigOnline_cfi")

process.load("L1TriggerConfig.DTTrackFinder.L1MuDTEtaPatternLutOnline_cfi")
process.load("L1TriggerConfig.DTTrackFinder.L1MuDTExtLutOnline_cfi")
process.load("L1TriggerConfig.DTTrackFinder.L1MuDTPhiLutOnline_cfi")
process.load("L1TriggerConfig.DTTrackFinder.L1MuDTPtaLutOnline_cfi")
process.load("L1TriggerConfig.DTTrackFinder.L1MuDTQualPatternLutOnline_cfi")
process.load("L1TriggerConfig.DTTrackFinder.L1MuDTTFParametersOnline_cfi")

process.load("L1TriggerConfig.RPCTriggerConfig.L1RPCConfigOnline_cfi")
process.load("L1TriggerConfig.RPCTriggerConfig.L1RPCConeDefinitionOnline_cfi")

process.load("L1TriggerConfig.GMTConfigProducers.L1MuGMTParametersOnlineProducer_cfi")
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerPtScaleOnlineProducer_cfi")
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerScalesOnlineProducer_cfi")
process.L1MuGMTParametersOnlineProducer.ignoreVersionMismatch = True

process.load("L1TriggerConfig.RCTConfigProducers.L1RCTParametersOnline_cfi")
process.load("L1TriggerConfig.L1ScalesProducers.L1EmEtScaleConfigOnline_cfi")

process.load("L1TriggerConfig.GctConfigProducers.L1GctJetFinderParamsOnline_cfi")
process.load("L1TriggerConfig.L1ScalesProducers.L1HtMissScaleOnline_cfi")
process.load("L1TriggerConfig.L1ScalesProducers.L1HfRingEtScaleOnline_cfi")
process.load("L1TriggerConfig.L1ScalesProducers.L1JetEtScaleOnline_cfi")

process.load("L1TriggerConfig.L1GtConfigProducers.l1GtParametersOnline_cfi")
process.load("L1TriggerConfig.L1GtConfigProducers.l1GtPsbSetupOnline_cfi")
process.load("L1TriggerConfig.L1GtConfigProducers.l1GtTriggerMenuOnline_cfi")

# writer modules
from CondTools.L1Trigger.L1CondDBPayloadWriter_cff import initPayloadWriter
initPayloadWriter( process,
                   outputDBConnect = options.outputDBConnect,
                   outputDBAuth = options.outputDBAuth,
                   tagBase = options.tagBase )
initPayloadWriter.outputDB.logconnect = cms.untracked.string('sqlite_file:o2o_payload_log.db')
process.L1CondDBPayloadWriter.logTransactions = True

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(4294967295),
    lastValue = cms.uint64(4294967295),
    interval = cms.uint64(1)
)

process.outputDB = cms.ESSource("PoolDBESSource",
    process.CondDBCommon,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('L1TriggerKeyListRcd'),
        tag = cms.string( "L1TriggerKeyList_" + options.tagBase )
    ),
                      cms.PSet(
    record = cms.string('L1GtStableParametersRcd'),
    tag = cms.string( "L1GtStableParameters_" + options.tagBase )
    ),
                      cms.PSet(
    record = cms.string('L1CaloGeometryRecord'),
    tag = cms.string( "L1CaloGeometry_" + options.tagBase )
    ))
)
process.outputDB.connect = cms.string(options.outputDBConnect)
process.outputDB.DBParameters.authenticationPath = options.outputDBAuth

process.p = cms.Path(process.L1CondDBPayloadWriter)
