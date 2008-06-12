import FWCore.ParameterSet.Config as cms

process = cms.Process("L1ConfigWriteIOVDummy")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

# Generate dummy L1TriggerKey but not L1TriggerKeyList
process.load("CondTools.L1Trigger.L1TriggerKeyDummy_cff")

# Get L1TriggerKeyList from DB
process.load("CondCore.DBCommon.CondDBCommon_cfi")

# writer modules
process.load("CondTools.L1Trigger.L1CondDBIOVWriter_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptyIOVSource",
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    timetype = cms.string('runnumber'),
    interval = cms.uint64(1)
)

process.orcon = cms.ESSource("PoolDBESSource",
    process.CondDBCommon,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('L1TriggerKeyListRcd'),
        tag = cms.string('L1TriggerKeyList_CRUZET_hlt')
    ))
)

process.p = cms.Path(process.L1CondDBIOVWriter)
process.orcon.connect = cms.string('sqlite_file:l1config.db')
process.orcon.DBParameters.authenticationPath = '.'


