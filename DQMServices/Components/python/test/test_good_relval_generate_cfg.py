import FWCore.ParameterSet.Config as cms

process = cms.Process("TestDQMFileSaver")
process.load("DQMServices.Components.test.MessageLogger_cfi")

process.load("Validation.GlobalHits.globalhits_tester_cff")

process.load("DQMServices.Components.MEtoEDMConverter_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
process.source = cms.Source("EmptySource")

process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *_MEtoEDMConverter_*_*'),
    fileName = cms.untracked.string('test_relval_generate.root')
)

process.p1 = cms.Path(process.globalhitstest*process.MEtoEDMConverter)
process.outpath = cms.EndPath(process.FEVT)
process.schedule = cms.Schedule(process.p1,process.outpath)



