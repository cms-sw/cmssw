import FWCore.ParameterSet.Config as cms

process = cms.Process("L1ConfigWritePayloadDummy")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.enable = cms.untracked.bool(True)
process.MessageLogger.cout.threshold = cms.untracked.string('INFO')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

# Generate dummy L1TriggerKeyList
#process.load("CondTools.L1Trigger.L1TriggerKeyListDummy_cff")

# Get configuration data from OMDS.  This is the subclass of L1ConfigOnlineProdBase.
process.load("L1TriggerConfig.RCTConfigProducers.L1RCTParametersOnline_cfi")
process.load("L1TriggerConfig.L1ScalesProducers.L1EmEtScaleConfigOnline_cfi")
process.load("L1TriggerConfig.L1ScalesProducers.L1CaloEcalScaleConfigOnline_cfi")
process.load("L1TriggerConfig.L1ScalesProducers.L1CaloHcalScaleConfigOnline_cfi")
process.l1RCTParametersTest = cms.EDAnalyzer("L1RCTParametersTester")
process.l1RCTChannelMaskTest = cms.EDAnalyzer("L1RCTChannelMaskTester")
process.l1scalesTest = cms.EDAnalyzer("L1ScalesTester")
# paths to be run





process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource")

from CondTools.L1Trigger.L1CondDBPayloadWriter_cff import initPayloadWriter
initPayloadWriter( process )




process.getter = cms.EDAnalyzer("EventSetupRecordDataGetter",
   toGet = cms.VPSet(cms.PSet(
   record = cms.string('L1EmEtScaleRcd'),
   data = cms.vstring('L1CaloEtScale')),
                     cms.PSet(
    record = cms.string('L1RCTParametersRcd'),
    data = cms.vstring('L1RCTParameters')
    ),
                     cms.PSet(
      record = cms.string('L1CaloHcalScaleRcd'),
    data = cms.vstring('L1CaloHcalScale')
    ),
                     cms.PSet(
    record = cms.string('L1CaloEcalScaleRcd'),
    data = cms.vstring('L1CaloEcalScale')
    )),
   verbose = cms.untracked.bool(True)
)

process.p = cms.Path(process.getter*process.l1scalesTest)

process.load("CondTools.L1Trigger.L1TriggerKeyDummy_cff")
process.L1TriggerKeyDummy.objectKeys = cms.VPSet()
process.L1TriggerKeyDummy.label = cms.string('SubsystemKeysOnly')

# rctKey = csctfKey, dttfKey, rpcKey, gmtKey, rctKey, gctKey, gtKey, or tsp0Key
#process.L1TriggerKeyDummy.rctKey = cms.string('EEG_EHSUMS_TAU3_DECO_25_FALLGR09_FULLECAL')
#process.L1TriggerKeyDummy.rctKey = cms.string('HEG_HSUMS_HF');
process.L1TriggerKeyDummy.rctKey = cms.string('EEG_EHSUMS_TAU4_NOVETOES_14APR10')
#process.L1TriggerKeyDummy.rctKey = cms.string('EE+');

# Subclass of L1ObjectKeysOnlineProdBase.
process.load("L1TriggerConfig.RCTConfigProducers.L1RCTObjectKeysOnline_cfi")
process.L1RCTObjectKeysOnline.subsystemLabel = cms.string('')

