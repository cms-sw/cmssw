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
#process.load("L1TriggerConfig.CSCTFConfigProducers.CSCTFAlignmentOnline_cfi")
process.load("L1TriggerConfig.CSCTFConfigProducers.L1MuCSCPtLutConfigOnline_cfi")

process.load("L1TriggerConfig.DTTrackFinder.L1MuDTEtaPatternLutOnline_cfi")
process.load("L1TriggerConfig.DTTrackFinder.L1MuDTExtLutOnline_cfi")
process.load("L1TriggerConfig.DTTrackFinder.L1MuDTPhiLutOnline_cfi")
process.load("L1TriggerConfig.DTTrackFinder.L1MuDTPtaLutOnline_cfi")
process.load("L1TriggerConfig.DTTrackFinder.L1MuDTQualPatternLutOnline_cfi")
process.load("L1TriggerConfig.DTTrackFinder.L1MuDTTFParametersOnline_cfi")

process.load("L1TriggerConfig.RPCTriggerConfig.L1RPCConfigOnline_cfi")
process.load("L1TriggerConfig.RPCTriggerConfig.L1RPCConeDefinitionOnline_cfi")
process.load("L1TriggerConfig.RPCTriggerConfig.L1RPCBxOrConfigOnline_cfi")
process.load("L1TriggerConfig.RPCTriggerConfig.L1RPCHsbConfigOnline_cfi")

if options.copyNonO2OPayloads == 1:
    process.L1MuDTEtaPatternLutOnline.copyFromCondDB = cms.bool( True )
    process.L1MuDTEtaPatternLutOnline.onlineDB = options.copyDBConnect
    process.L1MuDTEtaPatternLutOnline.onlineAuthentication = options.copyDBAuth
    process.L1MuDTExtLutOnline.copyFromCondDB = cms.bool( True )
    process.L1MuDTExtLutOnline.onlineDB = options.copyDBConnect
    process.L1MuDTExtLutOnline.onlineAuthentication = options.copyDBAuth
    process.L1MuDTPhiLutOnline.copyFromCondDB = cms.bool( True )
    process.L1MuDTPhiLutOnline.onlineDB = options.copyDBConnect
    process.L1MuDTPhiLutOnline.onlineAuthentication = options.copyDBAuth
    process.L1MuDTPtaLutOnline.copyFromCondDB = cms.bool( True )
    process.L1MuDTPtaLutOnline.onlineDB = options.copyDBConnect
    process.L1MuDTPtaLutOnline.onlineAuthentication = options.copyDBAuth
    process.L1MuDTQualPatternLutOnline.copyFromCondDB = cms.bool( True )
    process.L1MuDTQualPatternLutOnline.onlineDB = options.copyDBConnect
    process.L1MuDTQualPatternLutOnline.onlineAuthentication = options.copyDBAuth
    process.L1RPCConfigOnline.copyFromCondDB = cms.bool( True )
    process.L1RPCConfigOnline.onlineDB = options.copyDBConnect
    process.L1RPCConfigOnline.onlineAuthentication = options.copyDBAuth
    process.L1RPCConeDefinitionOnline.copyFromCondDB = cms.bool( True )
    process.L1RPCConeDefinitionOnline.onlineDB = options.copyDBConnect
    process.L1RPCConeDefinitionOnline.onlineAuthentication = options.copyDBAuth
    process.L1RPCHsbConfigOnline.copyFromCondDB = cms.bool( True )
    process.L1RPCHsbConfigOnline.onlineDB = options.copyDBConnect
    process.L1RPCHsbConfigOnline.onlineAuthentication = options.copyDBAuth
    process.L1RPCBxOrConfigOnline.copyFromCondDB = cms.bool( True )
    process.L1RPCBxOrConfigOnline.onlineDB = options.copyDBConnect
    process.L1RPCBxOrConfigOnline.onlineAuthentication = options.copyDBAuth

process.load("L1TriggerConfig.GMTConfigProducers.L1MuGMTParametersOnlineProducer_cfi")
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerPtScaleOnlineProducer_cfi")
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerScalesOnlineProducer_cfi")
process.L1MuGMTParametersOnlineProducer.ignoreVersionMismatch = True

process.load("L1TriggerConfig.RCTConfigProducers.L1RCTParametersOnline_cfi")
process.load("L1TriggerConfig.L1ScalesProducers.L1EmEtScaleConfigOnline_cfi")
process.load("L1TriggerConfig.L1ScalesProducers.L1CaloEcalScaleConfigOnline_cfi")
process.load("L1TriggerConfig.L1ScalesProducers.L1CaloHcalScaleConfigOnline_cfi")

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

if options.logTransactions == 1:
    initPayloadWriter.outputDB.logconnect = cms.untracked.string('oracle://cms_orcon_prod/CMS_COND_31X_POPCONLOG')
    process.L1CondDBPayloadWriter.logTransactions = True

if options.overwriteKeys == 0:
    process.L1CondDBPayloadWriter.overwriteKeys = False
else:
    process.L1CondDBPayloadWriter.overwriteKeys = True
                
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

# Suppress warnings, not actually used, except for copyNonO2OPayloads
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
                            
process.p = cms.Path(process.L1CondDBPayloadWriter)
