import FWCore.ParameterSet.Config as cms

process = cms.Process("TestDQMFileExtract")
process.load("DQMServices.Components.test.MessageLogger_cfi")

process.load("DQMServices.Components.EDMtoMEConverter_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:test_relval_generate.root')
)

process.p1 = cms.Path(process.EDMtoMEConverter*process.dqmSaver)
process.DQMStore.referenceFileName = 'Relval.Ref.root'
process.dqmSaver.referenceHandling = 'all'
process.dqmSaver.convention = 'RelVal'
process.dqmSaver.workflow = '/TestRelVal/XYZZY/RECO'


