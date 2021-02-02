import FWCore.ParameterSet.Config as cms

process = cms.Process("L1ConfigWritePayloadOnline")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.enable = cms.untracked.bool(True)
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

# Generate L1TriggerKey from OMDS
process.load("CondTools.L1Trigger.L1SubsystemKeysOnline_cfi")
process.L1SubsystemKeysOnline.tscKey = cms.string( options.tscKey )
process.load("CondTools.L1Trigger.L1ConfigTSCKeys_cff")
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
process.load("CondTools.L1Trigger.L1ConfigTSCPayloads_cff")

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

# Define CondDB tags
from CondTools.L1Trigger.L1CondEnum_cfi import L1CondEnum
from CondTools.L1Trigger.L1O2OTags_cfi import initL1O2OTags
initL1O2OTags()

# writer modules
from CondTools.L1Trigger.L1CondDBPayloadWriter_cff import initPayloadWriter
initPayloadWriter( process,
                   outputDBConnect = options.outputDBConnect,
                   outputDBAuth = options.outputDBAuth,
                   tagBaseVec = initL1O2OTags.tagBaseVec )

if options.logTransactions == 1:
#    initPayloadWriter.outputDB.logconnect = cms.untracked.string('oracle://cms_orcon_prod/CMS_COND_31X_POPCONLOG')
    initPayloadWriter.outputDB.logconnect = cms.untracked.string('sqlite_file:l1o2o-log.db')
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
    tag = cms.string( "L1TriggerKeyList_" + initL1O2OTags.tagBaseVec[ L1CondEnum.L1TriggerKeyList ] )
    ),
                                                  cms.PSet(
    record = cms.string('L1GtStableParametersRcd'),
    tag = cms.string( "L1GtStableParameters_" + initL1O2OTags.tagBaseVec[ L1CondEnum.L1GtStableParameters ] )
    ),
                                                  cms.PSet(
    record = cms.string('L1CaloGeometryRecord'),
    tag = cms.string( "L1CaloGeometry_" + initL1O2OTags.tagBaseVec[ L1CondEnum.L1CaloGeometry ] )
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
                            
process.p = cms.Path(process.L1CondDBPayloadWriter)
