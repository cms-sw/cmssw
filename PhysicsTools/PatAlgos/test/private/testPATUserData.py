import FWCore.ParameterSet.Config as cms

process = cms.Process("TestUserData")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:patLayer1.root')
)

process.MessageLogger = cms.Service("MessageLogger")

process.testWrite = cms.EDProducer("PATUserDataTestModule",
    mode  = cms.string("write"),
    muons = cms.InputTag("selectedLayer1Muons"),
)

process.testRead = cms.EDProducer("PATUserDataTestModule",
    mode  = cms.string("read"),
    muons = cms.InputTag("testWrite"),
)


process.p = cms.Path(
    process.testWrite +
    process.testRead
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('layer1muons_withUserData.root'),
    outputCommands = cms.untracked.vstring("drop *", "keep patMuons_*_*_*"),
)
process.out_step = cms.EndPath(process.out)
