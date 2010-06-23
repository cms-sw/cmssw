
import FWCore.ParameterSet.Config as cms

# The top-level process
process = cms.Process("TEST")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("Configuration.StandardSequences.Services_cff")
process.load("L1Trigger.GlobalCaloTrigger.test.gctConfig_cff")

## Source
process.source = cms.Source("EmptySource")

# the printLuts module
process.load("L1Trigger.GlobalCaloTrigger.l1GctPrintLuts_cfi")

process.maxEvents = cms.untracked.PSet ( input = cms.untracked.int32(1) )

process.p = cms.Path(process.l1GctPrintLuts)

