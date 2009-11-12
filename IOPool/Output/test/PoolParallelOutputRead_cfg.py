import FWCore.ParameterSet.Config as cms

process = cms.Process("TESTOUTPUTREAD")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:PoolOutputTest_0.root','file:PoolOutputTest_1.root','file:PoolOutputTest_2.root')
)



