import FWCore.ParameterSet.Config as cms

process = cms.Process("L1ConfigWritePayloadDummy")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

# Generate dummy L1TriggerKey and L1TriggerKeyList
process.load("CondTools.L1Trigger.L1TriggerKeyDummy_cff")

process.load("CondTools.L1Trigger.L1TriggerKeyListDummy_cff")

# Generate dummy configuration data
process.load("L1Trigger.Configuration.L1DummyConfig_cff")

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

process.p = cms.Path(process.L1CondDBPayloadWriter)
process.l1CSCTFConfig.ptLUT_path = '/afs/cern.ch/cms/MUON/csc/fast1/track_finder/luts/PtLUT.dat'


