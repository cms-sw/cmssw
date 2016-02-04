import FWCore.ParameterSet.Config as cms

process = cms.Process("TestMEtoEDMConverter")
process.load("DQMServices.Examples.test.MessageLogger_cfi")

process.load("DQMServices.Examples.test.ConverterTester_cff")

process.load("DQMServices.Components.MEtoEDMConverter_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *_MEtoEDMConverter_*_*'),
    fileName = cms.untracked.string('MEtoEDMConverter.root')
)

process.p1 = cms.Path(process.convertertester*process.MEtoEDMConverter)
process.outpath = cms.EndPath(process.FEVT)
process.schedule = cms.Schedule(process.p1,process.outpath)

process.MEtoEDMConverter.MEPathToSave = ''

