# cfg file to test the setup of L1 GT PSB boards.

import FWCore.ParameterSet.Config as cms

# process
process = cms.Process("L1GtPsbSetupTest")
process.l1GtPsbSetupTest = cms.EDAnalyzer("L1GtPsbSetupTester")

# number of events and source
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource")

# configuration
process.load("L1TriggerConfig.L1GtConfigProducers.L1GtPsbSetupConfig_cff")

# path to be run
process.p = cms.Path(process.l1GtPsbSetupTest)

