import FWCore.ParameterSet.Config as cms

process = cms.Process("MERGE")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
'file:small.root',
'file:big.root'
)
)



