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

##  Use the two lines below to test the output of L1 with embedded external user data
#process.source.fileNames = cms.untracked.vstring('file:PATLayer1_Output.fromAOD_full.root');
#process.testRead.muons = cms.InputTag("selectedLayer1Muons")

process.p = cms.Path(
    process.testRead
)
