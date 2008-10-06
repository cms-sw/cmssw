import FWCore.ParameterSet.Config as cms

process = cms.Process("L1ConfigWritePayloadOnline")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.placeholder = cms.untracked.bool(False)
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

process.load("CondCore.DBCommon.CondDBCommon_cfi")

# Generate L1TriggerKey and configuration data from OMDS
process.load("CondTools.L1Trigger.L1SubsystemKeysOnline_cfi")
process.L1SubsystemKeysOnline.tscKey = cms.string( 'TSC_CRUZET2_080613_GTmuon_GMTDTRPC5CSC5_CSCclosedwindow_DTTFtopbot_RPC_LUM_GCT_RCTH' )

process.load("L1TriggerConfig.RCTConfigProducers.L1RCTObjectKeysOnline_cfi")

process.load("CondTools.L1Trigger.L1TriggerKeyOnline_cfi")
process.L1TriggerKeyOnline.subsystemLabels = cms.vstring( 'RCT' )

process.load("L1TriggerConfig.RCTConfigProducers.L1RCTConfigOnline_cfi")

# writer modules
process.load("CondTools.L1Trigger.L1CondDBPayloadWriter_cfi")
process.L1CondDBPayloadWriter.offlineDB = cms.string('sqlite_file:l1config.db')
#process.L1CondDBPayloadWriter.offlineDB = cms.string('oracle://cms_orcon_prod/CMS_COND_21X_L1T')
#process.L1CondDBPayloadWriter.offlineAuthentication = '/nfshome0/xiezhen/conddb'

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
        tag = cms.string('L1TriggerKeyList_IDEAL')
    ))
)
process.orcon.connect = cms.string('sqlite_file:l1config.db')
#process.orcon.connect = cms.string('oracle://cms_orcon_prod/CMS_COND_21X_L1T')
#process.orcon.DBParameters.authenticationPath = '/nfshome0/xiezhen/conddb'

process.p = cms.Path(process.L1CondDBPayloadWriter)


