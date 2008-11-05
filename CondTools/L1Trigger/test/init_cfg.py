import FWCore.ParameterSet.Config as cms

process = cms.Process("L1ConfigWritePayloadDummy")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.placeholder = cms.untracked.bool(False)
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    interval = cms.uint64(1)
)

# Generate dummy L1TriggerKey and L1TriggerKeyList
process.load("CondTools.L1Trigger.L1TriggerKeyDummy_cff")
process.L1TriggerKeyDummy.objectKeys = cms.VPSet()
process.L1TriggerKeyDummy.tscKey = cms.string('')

# Use dummy producer to initialize DB on the first time ONLY.
process.load("CondTools.L1Trigger.L1TriggerKeyListDummy_cff")

# writer modules
process.load("CondTools.L1Trigger.L1CondDBPayloadWriter_cfi")
process.L1CondDBPayloadWriter.L1TriggerKeyListTag = cms.string('L1TriggerKeyList_IDEAL')
#process.L1CondDBPayloadWriter.offlineDB = cms.string('oracle://cms_orcon_prod/CMS_COND_21X_L1T')
#process.L1CondDBPayloadWriter.offlineAuthentication = cms.string('/nfshome0/xiezhen/conddb')
process.L1CondDBPayloadWriter.offlineDB = cms.string('sqlite_file:l1config.db')
process.L1CondDBPayloadWriter.offlineAuthentication = cms.string('.')

process.p = cms.Path(process.L1CondDBPayloadWriter)
