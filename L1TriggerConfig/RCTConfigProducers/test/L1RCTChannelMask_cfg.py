# cfg file to test L1 RCT parameters

import FWCore.ParameterSet.Config as cms

# process
process = cms.Process("L1RCTChannelMaskTest")
process.l1RCTChannelMaskTest = cms.EDAnalyzer("L1RCTChannelMaskTester")

# number of events and source
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource")

# configuration
process.load("L1TriggerConfig.RCTConfigProducers.L1RCTConfig_cff")

# path to be run
process.p = cms.Path(process.l1RCTChannelMaskTest)



