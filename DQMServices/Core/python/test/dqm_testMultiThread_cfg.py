import FWCore.ParameterSet.Config as cms

process = cms.Process("DQMMULTITHREAD")
process.load("DQMServices.Core.DQM_cfg")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.source = cms.Source("EmptySource")

process.dqm_multi_thread = cms.EDAnalyzer("DQMTestMultiThread")

process.p = cms.Path(process.dqm_multi_thread)

process.Tracer = cms.Service('Tracer')
