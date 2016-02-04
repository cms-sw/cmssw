import FWCore.ParameterSet.Config as cms
process = cms.Process("READ")
process.read = cms.EDAnalyzer(
    "TestFindProduct",
    inputTags =cms.untracked.VInputTag(cms.InputTag("i")),
    expectedSum = cms.untracked.int32(0)
)
process.p = cms.Path(process.read)
process.options = cms.untracked.PSet(Rethrow=cms.untracked.vstring("ProductNotFound"))

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring("file:unscheduled_fail_on_output.root")
)