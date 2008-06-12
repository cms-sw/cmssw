import FWCore.ParameterSet.Config as cms

process = cms.Process("L1ConfigWritePayloadOnline")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("CondCore.DBCommon.CondDBCommon_cfi")

# I don't think these are needed...
# include "CondTools/L1Trigger/data/L1SubsystemParams.cfi"
# replace orcon.toGet += L1SubsystemParams.recordInfo
# Generate L1TriggerKey and configuration data from OMDS
process.load("CondTools.L1Trigger.L1TriggerKeyOnline_cfi")

process.load("CondTools.L1Trigger.L1TriggerConfigOnline_cfi")

# writer modules
process.load("CondTools.L1Trigger.L1CondDBPayloadWriter_cfi")

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
    ), 
        cms.PSet(
            record = cms.string('L1TriggerKeyRcd'),
            tag = cms.string('L1TriggerKey_CRUZET_hlt')
        ))
)

process.p = cms.Path(process.L1CondDBPayloadWriter)
process.orcon.connect = cms.string('sqlite_file:l1config.db')
process.orcon.DBParameters.authenticationPath = '.'


