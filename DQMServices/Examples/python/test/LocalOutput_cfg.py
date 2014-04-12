import FWCore.ParameterSet.Config as cms

process = cms.Process("TestMEtoEDMConverter")
process.load("DQMServices.Examples.test.MessageLogger_cfi")

process.load("DQMServices.Examples.test.ConverterTester_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.LoadAllDictionaries = cms.Service("LoadAllDictionaries")

process.p1 = cms.Path(process.convertertester*process.dqmSaver)

