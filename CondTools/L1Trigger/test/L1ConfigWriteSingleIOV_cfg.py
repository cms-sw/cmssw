import FWCore.ParameterSet.Config as cms

process = cms.Process("L1ConfigWriteIOVDummy")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.placeholder = cms.untracked.bool(False)
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

# Get L1TriggerKeyList from DB
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.orcon = cms.ESSource("PoolDBESSource",
    process.CondDBCommon,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('L1TriggerKeyListRcd'),
        tag = cms.string('L1TriggerKeyList_IDEAL')
    ))
)
process.orcon.connect = cms.string('sqlite_file:l1config.db')
process.orcon.DBParameters.authenticationPath = '.'

# writer modules
process.load("CondTools.L1Trigger.L1CondDBIOVWriter_cff")
process.L1CondDBIOVWriter.tscKey = cms.string('dummy')
process.L1CondDBIOVWriter.ignoreTriggerKey = cms.bool(True)
process.L1CondDBIOVWriter.toPut = cms.VPSet(cms.PSet(
    record = cms.string('L1RCTParametersRcd'),
    type = cms.string('L1RCTParameters'),
    tag = cms.string('L1RCTParameters_IDEAL')
))

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1000),
    lastValue = cms.uint64(1000),
    interval = cms.uint64(1)
)

process.p = cms.Path(process.L1CondDBIOVWriter)


