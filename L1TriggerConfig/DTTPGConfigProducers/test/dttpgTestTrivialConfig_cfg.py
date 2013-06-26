import FWCore.ParameterSet.Config as cms

process = cms.Process("L1DTTrigProd")

process.load("L1TriggerConfig.DTTPGConfigProducers.L1DTTPGConfig_cff")

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(3),
    firstRun = cms.untracked.uint32(147000)
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(9)
)

process.dumpConfig = cms.EDAnalyzer("DTConfigTester",
    wheel   = cms.untracked.int32(0),
    sector  = cms.untracked.int32(4),
    station = cms.untracked.int32(1),
    traco = cms.untracked.int32(2),
    bti = cms.untracked.int32(9)
)

process.p = cms.Path(process.dumpConfig)

