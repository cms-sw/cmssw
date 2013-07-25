# cfg file to test the mappings of the L1 GT boards

import FWCore.ParameterSet.Config as cms

# process
process = cms.Process("L1GtBoardMapsTest")
process.l1GtBoardMapsTest = cms.EDAnalyzer("L1GtBoardMapsTester")

# number of events and source
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource")

# configuration
process.load("L1TriggerConfig.L1GtConfigProducers.L1GtBoardMapsConfig_cff")

# path to be run
process.p = cms.Path(process.l1GtBoardMapsTest)

