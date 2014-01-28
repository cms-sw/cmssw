# cfg file to test L1 GT prescale factors and trigger mask

import FWCore.ParameterSet.Config as cms

# process
process = cms.Process("L1GtPrescaleFactorsAndMasksTest")
process.l1GtPrescaleFactorsAndMasksTest = cms.EDAnalyzer("L1GtPrescaleFactorsAndMasksTester")

# number of events and source
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource")

# configuration
process.load("L1TriggerConfig.L1GtConfigProducers.L1GtPrescaleFactorsAlgoTrigConfig_cff")
process.load("L1TriggerConfig.L1GtConfigProducers.L1GtPrescaleFactorsTechTrigConfig_cff")
process.load("L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskAlgoTrigConfig_cff")
process.load("L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff")
process.load("L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskVetoAlgoTrigConfig_cff")
process.load("L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskVetoTechTrigConfig_cff")

# path to be run
process.p = cms.Path(process.l1GtPrescaleFactorsAndMasksTest)

