import FWCore.ParameterSet.Config as cms

process = cms.Process("TestUserDataRead")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:layer1muons_withUserData.root')
)

process.MessageLogger = cms.Service("MessageLogger")

process.testRead = cms.EDProducer("PATUserDataTestModule",
    mode  = cms.string("read"),
    muons = cms.InputTag("testWrite"),
)


process.p = cms.Path(
    process.testRead
)
