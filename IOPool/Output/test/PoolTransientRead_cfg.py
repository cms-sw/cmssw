import FWCore.ParameterSet.Config as cms

process = cms.Process("TESTTRANSIENTREAD")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:PoolTransientTest.root')
)



# foo bar baz
# Fd5d0OliFaFCk
