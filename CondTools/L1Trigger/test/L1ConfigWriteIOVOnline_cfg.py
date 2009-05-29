import FWCore.ParameterSet.Config as cms

process = cms.Process("L1ConfigWriteIOVOnline")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.placeholder = cms.untracked.bool(False)
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

# Get L1TriggerKeyList from DB
process.load("CondCore.DBCommon.CondDBCommon_cfi")

# Generate TSC key
process.load("CondTools.L1Trigger.L1TriggerKeyDummy_cff")

# writer modules
process.load("CondTools.L1Trigger.L1CondDBIOVWriter_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstRun = cms.untracked.uint32(1),
    lastRun = cms.untracked.uint32(1),
    interval = cms.uint32(1)
)

process.orcon = cms.ESSource("PoolDBESSource",
    process.CondDBCommon,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('L1TriggerKeyListRcd'),
        tag = cms.string('L1TriggerKeyList_IDEAL')
    ))
)

process.p = cms.Path(process.L1CondDBIOVWriter)
process.orcon.connect = cms.string('oracle://cms_orcon_prod/CMS_COND_21X_L1T')
process.orcon.DBParameters.authenticationPath = '/nfshome0/onlinedbadm/conddb'
process.L1CondDBIOVWriter.offlineDB = cms.string('oracle://cms_orcon_prod/CMS_COND_21X_L1T')
process.L1CondDBIOVWriter.offlineAuthentication = '/nfshome0/onlinedbadm/conddb'


