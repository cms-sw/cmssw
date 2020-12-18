# cfg file to test the online producer of L1GtParametersRcd

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

# TSC key
process.load("CondTools.L1Trigger.L1SubsystemKeysOnline_cfi")
process.L1SubsystemKeysOnline.tscKey = cms.string( 'TSC_000610_081121_CRAFT_GTgtstartupbase3tm2v2_GMTstartupdt_GCT_RCT_DTTF_ECAL_HCAL_DT_MI' )

# Subclass of L1ObjectKeysOnlineProdBase.
process.load("L1TriggerConfig.L1GtConfigProducers.l1GtTscObjectKeysOnline_cfi")
process.l1GtTscObjectKeysOnline.subsystemLabel = cms.string('')

# Generate dummy L1TriggerKeyList
process.load("CondTools.L1Trigger.L1TriggerKeyListDummy_cff")

# Get configuration data from OMDS.  This is the subclass of L1ConfigOnlineProdBase.
process.load("L1TriggerConfig.L1GtConfigProducers.l1GtParametersOnline_cfi")


process.getter = cms.EDAnalyzer("EventSetupRecordDataGetter",
   toGet = cms.VPSet(cms.PSet(
   record = cms.string('L1GtParametersRcd'),
   data = cms.vstring('L1GtParameters')
   )),
   verbose = cms.untracked.bool(True)
)

process.p = cms.Path(process.getter)

# Message Logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.enable = cms.untracked.bool(True)
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

