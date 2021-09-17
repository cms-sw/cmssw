import FWCore.ParameterSet.Config as cms

# This is to write a single payload of L1RPCConfig to the DB using the L1 O2O.
# CondTools/L1Trigger/test/init_cfg.py must be run first to initialize the
# L1TriggerKeyList in the DB, then L1RPCConfig payloads can be written one by one.

process = cms.Process("L1ConfigWritePayloadDummy")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.enable = cms.untracked.bool(True)
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

# Generate L1TriggerKey - here define the RPC key
process.load("CondTools.L1Trigger.L1TriggerKeyDummy_cff")
process.L1TriggerKeyDummy.objectKeys = cms.VPSet(cms.PSet(
    record = cms.string('L1RPCConfigRcd'),
    type = cms.string('L1RPCConfig'),
    key = cms.string('DEFAULT')
#    key = cms.string('COSMIC')
))

# Get L1TriggerKeyList from DB
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.orcon = cms.ESSource("PoolDBESSource",
    process.CondDBCommon,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('L1TriggerKeyListRcd'),
        tag = cms.string('L1TriggerKeyList_IDEAL')
    ))
)
process.es_prefer_orcon = cms.ESPrefer("PoolDBESSource","orcon")
process.orcon.connect = cms.string('sqlite_file:l1config.db')
process.orcon.DBParameters.authenticationPath = cms.untracked.string('.')

# Generate L1RPCConfig object - here define the payload associated to the RPC key
process.rpcconf = cms.ESProducer("RPCTriggerConfig",
#    filedir = cms.untracked.string('L1Trigger/RPCTrigger/data/Eff90PPT12/'),
#    filedir = cms.untracked.string('L1Trigger/RPCTrigger/data/CosmicPats6/'),
    filedir = cms.untracked.string('../../CR0T/CosmicPats6/'),
#    PACsPerTower = cms.untracked.int32(12)
    PACsPerTower = cms.untracked.int32(1)
)
process.rpcconfsrc = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1RPCConfigRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

# writer modules
process.load("CondTools.L1Trigger.L1CondDBPayloadWriter_cff")
process.L1CondDBPayloadWriter.writeL1TriggerKey = cms.bool(False)
process.L1CondDBPayloadWriter.L1TriggerKeyListTag = cms.string('L1TriggerKeyList_IDEAL')
process.L1CondDBPayloadWriter.offlineDB = cms.string('sqlite_file:l1config.db')
process.L1CondDBPayloadWriter.offlineAuthentication = cms.string('.')

# Use highest possible run number so we always get the latest version
# of L1TriggerKeyList.
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(4294967295),
    lastValue = cms.uint64(4294967295),
    interval = cms.uint64(1)
)

process.p = cms.Path(process.L1CondDBPayloadWriter)

