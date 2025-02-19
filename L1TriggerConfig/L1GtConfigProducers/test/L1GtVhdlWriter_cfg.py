# cfg file to write the VHDL templates

import FWCore.ParameterSet.Config as cms

# process
process = cms.Process("L1GtVhdlWriterTest")

# number of events and source
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource")

# configuration
process.load("L1TriggerConfig.L1GtConfigProducers.L1GtConfig_cff")

process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1x1032.L1Menu2007_cff")
#process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1030.L1Menu2008_2E30_cff")
#process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1031.L1Menu2008_2E31_cff")
#process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1x1032.L1MenuTestCondCorrelation_cff")

process.load("L1TriggerConfig.L1GtConfigProducers.l1GtVhdlWriter_cfi")

# path to be run
process.p = cms.Path(process.l1GtVhdlWriter)

# services

# Message Logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.debugModules = ['l1GtVhdlWriterTest']
process.MessageLogger.cout = cms.untracked.PSet(
    INFO = cms.untracked.PSet(
        limit = cms.untracked.int32(-1)
    ),
    threshold = cms.untracked.string('DEBUG'), ## DEBUG 

    DEBUG = cms.untracked.PSet( ## DEBUG, all messages  

        limit = cms.untracked.int32(-1)
    )
)

