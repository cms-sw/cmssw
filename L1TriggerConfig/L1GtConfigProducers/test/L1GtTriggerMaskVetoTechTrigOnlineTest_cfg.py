# cfg file to test the online producer of L1GtTriggerMaskVetoTechTrigRcd

import FWCore.ParameterSet.Config as cms

process = cms.Process("L1ConfigWritePayloadDummy")

# number of events and source
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    interval = cms.uint64(1)
)

# Generate dummy L1TriggerKeyList
process.load("CondTools.L1Trigger.L1TriggerKeyListDummy_cff")

# Get configuration data from OMDS.  This is the subclass of L1ConfigOnlineProdBase.

# key for partition X (default: 0)
partitionNr = cms.int32(0)

process.load("L1TriggerConfig.L1GtConfigProducers.L1GtRsObjectKeysOnline_cff")
process.l1GtRsObjectKeysOnline.PartitionNumber = partitionNr 

process.load("L1TriggerConfig.L1GtConfigProducers.l1GtTriggerMaskVetoTechTrigOnline_cfi")
process.l1GtTriggerMaskVetoTechTrigOnline.PartitionNumber = partitionNr 

process.getter = cms.EDAnalyzer("EventSetupRecordDataGetter",
   toGet = cms.VPSet(cms.PSet(
   record = cms.string('L1GtTriggerMaskVetoTechTrigRcd'),
   data = cms.vstring('L1GtTriggerMask')
   )),
   verbose = cms.untracked.bool(True)
)

process.p = cms.Path(process.getter)

# Message Logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.enable = cms.untracked.bool(True)
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('*')
