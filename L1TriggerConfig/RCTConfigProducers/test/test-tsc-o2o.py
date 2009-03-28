import FWCore.ParameterSet.Config as cms

process = cms.Process("L1ConfigWritePayloadDummy")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.placeholder = cms.untracked.bool(False)
process.MessageLogger.cout.threshold = cms.untracked.string('INFO')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

# Generate dummy L1TriggerKeyList
process.load("CondTools.L1Trigger.L1TriggerKeyListDummy_cff")

# Get configuration data from OMDS.  This is the subclass of L1ConfigOnlineProdBase.
process.load("L1TriggerConfig.RCTConfigProducers.L1RCTParametersOnline_cfi")
process.load("L1TriggerConfig.RCTConfigProducers.L1EmEtScaleConfigOnline_cfi")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    interval = cms.uint64(1)
)



process.getter = cms.EDAnalyzer("EventSetupRecordDataGetter",
   toGet = cms.VPSet(cms.PSet(
   record = cms.string('L1EmEtScaleRcd'),
   data = cms.vstring('L1CaloEtScale')),
                     cms.PSet(
    record = cms.string('L1RCTParametersRcd'),
    data = cms.vstring('L1RCTParameters')
    )),
   verbose = cms.untracked.bool(True)
)

process.p = cms.Path(process.getter)

process.load("CondTools.L1Trigger.L1TriggerKeyDummy_cff")
process.L1TriggerKeyDummy.objectKeys = cms.VPSet()
process.L1TriggerKeyDummy.label = cms.string('SubsystemKeysOnly')

# rctKey = csctfKey, dttfKey, rpcKey, gmtKey, rctKey, gctKey, gtKey, or tsp0Key
process.L1TriggerKeyDummy.rctKey = cms.string('RCT_DUMMY')

# Subclass of L1ObjectKeysOnlineProdBase.
process.load("L1TriggerConfig.RCTConfigProducers.L1RCTObjectKeysOnline_cfi")
process.L1RCTObjectKeysOnline.subsystemLabel = cms.string('')

