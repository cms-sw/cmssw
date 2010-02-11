import FWCore.ParameterSet.Config as cms

import sys
import os

# arguments
if (len(sys.argv)>1) :
    key=str(sys.argv[2])
else :
    key='Default'

# CMSSW config
process = cms.Process("L1ConfigWritePayloadDummy")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.placeholder = cms.untracked.bool(False)
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('l1GctConfigDump')

# Generate dummy L1TriggerKeyList
process.load("CondTools.L1Trigger.L1TriggerKeyListDummy_cff")

# get 
process.load("CondTools.L1Trigger.L1TriggerKeyDummy_cff")
process.L1TriggerKeyDummy.objectKeys = cms.VPSet()
process.L1TriggerKeyDummy.label = cms.string('SubsystemKeysOnly')

# xxxKey = csctfKey, dttfKey, rpcKey, gmtKey, rctKey, gctKey, gtKey, or tsp0Key
process.L1TriggerKeyDummy.gctKey = cms.string(key)

# Subclass of L1ObjectKeysOnlineProdBase.
process.load("L1TriggerConfig.GctConfigProducers.L1GctTSCObjectKeysOnline_cfi")
process.L1GctTSCObjectKeysOnline.subsystemLabel = cms.string('')

# Get configuration data from OMDS.  This is the subclass of L1ConfigOnlineProdBase.
process.load("L1TriggerConfig.GctConfigProducers.L1GctJetFinderParamsOnline_cfi")
process.load("L1TriggerConfig.L1ScalesProducers.L1JetEtScaleOnline_cfi")
process.load("L1TriggerConfig.L1ScalesProducers.L1HfRingEtScaleOnline_cfi")
process.load("L1TriggerConfig.L1ScalesProducers.L1HtMissScaleOnline_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    interval = cms.uint64(1)
)

process.getter1 = cms.EDAnalyzer("EventSetupRecordDataGetter",
   toGet = cms.VPSet(cms.PSet(
   record = cms.string('L1GctJetFinderParamsRcd'),
   data = cms.vstring('L1GctJetFinderParams')
   )),
   verbose = cms.untracked.bool(True)
)

process.getter2 = cms.EDAnalyzer("EventSetupRecordDataGetter",
   toGet = cms.VPSet(cms.PSet(
   record = cms.string('L1GctChannelMaskRcd'),
   data = cms.vstring('L1GctChannelMask')
   )),
   verbose = cms.untracked.bool(True)
)

process.getter3 = cms.EDAnalyzer("EventSetupRecordDataGetter",
   toGet = cms.VPSet(cms.PSet(
   record = cms.string('L1JetEtScaleRcd'),
   data = cms.vstring('L1CaloEtScale')
   )),
   verbose = cms.untracked.bool(True)
)

process.getter4 = cms.EDAnalyzer("EventSetupRecordDataGetter",
   toGet = cms.VPSet(cms.PSet(
   record = cms.string('L1HfRingEtScaleRcd'),
   data = cms.vstring('L1CaloEtScale')
   )),
   verbose = cms.untracked.bool(True)
)

process.getter5 = cms.EDAnalyzer("EventSetupRecordDataGetter",
   toGet = cms.VPSet(cms.PSet(
   record = cms.string('L1HtMissScaleRcd'),
   data = cms.vstring('L1CaloEtScale')
   )),
   verbose = cms.untracked.bool(True)
)

process.load('L1TriggerConfig/GctConfigProducers.l1GctConfigDump_cfi')

process.p = cms.Path(
#    process.getter1 +
#    process.getter2 +
#    process.getter3 +
#    process.getter4 +
#    process.getter5 +
    process.l1GctConfigDump
)
