import FWCore.ParameterSet.Config as cms

process = cms.Process("READ")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring("file:testExistingDictionaryChecking.root")
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.read = cms.EDAnalyzer("ExistingDictionaryTestAnalyzer",
    src = cms.InputTag("prod"),
    testVecUniqInt = cms.bool(False) # reading in plain vector<unique_ptr<int>> does not work yet
)

process.p = cms.Path(process.read)
