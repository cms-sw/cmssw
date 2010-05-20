
import FWCore.ParameterSet.Config as cms

# The top-level process
process = cms.Process("TEST")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("Configuration.StandardSequences.Services_cff")

## Source
process.source = cms.Source("EmptySource")

# include L1 emulator configuration
startupConfig = bool(True)

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
if startupConfig:
    process.GlobalTag.globaltag = cms.string('START3X_V20::All')
else:
    process.GlobalTag.globaltag = cms.string('MC_3XY_V20::All')

# process.load("L1Trigger.Configuration.L1StartupConfig_cff")
# but only part of it since I can't get a working set of tags
# that includes all the muon stuff
# L1 Calo configuration
#process.load("L1TriggerConfig.GctConfigProducers.L1GctConfig_cff")
#process.load("L1TriggerConfig.L1ScalesProducers.L1CaloScalesConfig_cff")
#process.load("L1TriggerConfig.L1GeometryProducers.l1CaloGeomConfig_cff")


# the printLuts module
process.load("L1Trigger.GlobalCaloTrigger.l1GctPrintLuts_cfi")

process.maxEvents = cms.untracked.PSet ( input = cms.untracked.int32(1) )

process.p = cms.Path(process.l1GctPrintLuts)

