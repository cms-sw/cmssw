import FWCore.ParameterSet.Config as cms

process = cms.Process("L1DTTrigProd")

process.load("L1TriggerConfig.DTTPGConfigProducers.L1DTTPGConfigFromDB_cff")

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(142089)
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.dumpConfig = cms.EDAnalyzer("DTConfigTester",
    wheel   = cms.untracked.int32(0),
    sector  = cms.untracked.int32(4),
    station = cms.untracked.int32(1),
    traco = cms.untracked.int32(2),
    bti = cms.untracked.int32(9)
)

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "GR_R_42_V6::All"

process.p = cms.Path(process.dumpConfig)

