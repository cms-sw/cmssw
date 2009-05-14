import FWCore.ParameterSet.Config as cms

process = cms.Process("L1ConfigWritePayloadDummy")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.placeholder = cms.untracked.bool(False)
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

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
    timetype = cms.string('runnumber'),
    firstRun = cms.untracked.uint32(1),
    lastRun = cms.untracked.uint32(1),
    interval = cms.uint32(1)
)

process.p = cms.Path(process.L1CondDBPayloadWriter)
process.l1CSCTFConfig.ptLUT_path = '/afs/cern.ch/cms/MUON/csc/fast1/track_finder/luts/PtLUT.dat'


