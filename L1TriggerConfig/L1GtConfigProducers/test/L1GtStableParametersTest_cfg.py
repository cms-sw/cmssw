# cfg file to test L1 GT stable parameters

import FWCore.ParameterSet.Config as cms

# process
process = cms.Process("L1GtStableParametersTest")
process.l1GtStableParametersTest = cms.EDAnalyzer("L1GtStableParametersTester")

# number of events and source
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource")

# configuration
process.load("L1TriggerConfig.L1GtConfigProducers.L1GtStableParametersConfig_cff")

# path to be run
process.p = cms.Path(process.l1GtStableParametersTest)

