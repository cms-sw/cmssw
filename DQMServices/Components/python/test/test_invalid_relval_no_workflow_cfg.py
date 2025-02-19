import FWCore.ParameterSet.Config as cms

process = cms.Process("TestDQMFileSaver1")
process.load("DQMServices.Components.test.MessageLogger_cfi")

process.load("DQMServices.Components.EDMtoMEConverter_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:test_relval_generate.root')
)

process.p1 = cms.Path(process.EDMtoMEConverter*process.dqmSaver)
process.dqmSaver.convention = 'RelVal'


