import FWCore.ParameterSet.Config as cms

process = cms.Process("L1ConfigWriteIOVOnline")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.placeholder = cms.untracked.bool(False)
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

# Get L1TriggerKeyList from DB
process.load("CondCore.DBCommon.CondDBCommon_cfi")

# writer modules
process.load("CondTools.L1Trigger.L1CondDBIOVWriter_cfi")
process.L1CondDBIOVWriter.offlineDB = cms.string('sqlite_file:l1config.db')
#process.L1CondDBIOVWriter.offlineDB = cms.string('oracle://cms_orcon_prod/CMS_COND_21X_L1T')
#process.L1CondDBIOVWriter.offlineAuthentication = '/nfshome0/xiezhen/conddb'
process.L1CondDBIOVWriter.tscKey = cms.string( 'TSC_CRUZET2_080613_GTmuon_GMTDTRPC5CSC5_CSCclosedwindow_DTTFtopbot_RPC_LUM_GCT_RCTH' )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstRun = cms.untracked.uint32(1000),
    lastRun = cms.untracked.uint32(1000),
    interval = cms.uint32(1)
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

process.p = cms.Path(process.L1CondDBIOVWriter)


