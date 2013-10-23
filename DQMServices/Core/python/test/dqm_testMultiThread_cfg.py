import FWCore.ParameterSet.Config as cms

process = cms.Process("DQMMULTITHREAD")
process.load("DQMServices.Core.DQM_cfg")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
)

process.source = cms.Source("EmptySource",
                            numberEventsInRun = cms.untracked.uint32(5),
                            firstLuminosityBlock = cms.untracked.uint32(1),
                            firstEvent = cms.untracked.uint32(1),
                            firstRun = cms.untracked.uint32(1),
                            numberEventsInLuminosityBlock = cms.untracked.uint32(1))

process.dqm_multi_thread = cms.EDAnalyzer("DQMTestMultiThread")

process.p = cms.Path(process.dqm_multi_thread)

#process.Tracer = cms.Service('Tracer')
