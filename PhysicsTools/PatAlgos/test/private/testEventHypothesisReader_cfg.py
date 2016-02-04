import FWCore.ParameterSet.Config as cms

process = cms.Process("TestEHR")
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:eh.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.read = cms.EDFilter("TestEventHypothesisReader",
    events = cms.InputTag("sols")
)

process.p = cms.Path(process.read)


