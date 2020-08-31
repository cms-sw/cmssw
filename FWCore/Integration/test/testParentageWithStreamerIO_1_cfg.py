import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST1")

process.source = cms.Source("EmptySource",
    firstEvent = cms.untracked.uint32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.prod1 = cms.EDProducer("AddIntsProducer",
    labels = cms.VInputTag()
)

process.prod2 = cms.EDProducer("AddIntsProducer",
    labels = cms.VInputTag("prod1")
)

process.test1 = cms.EDAnalyzer("TestParentage",
    inputTag = cms.InputTag("prod2"),
    expectedAncestors = cms.vstring("prod1")
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testParentageWithStreamerIO1.root'),
    outputCommands = cms.untracked.vstring(
        'keep *'
    ),
    dropMetaData = cms.untracked.string('ALL')
)

process.path1 = cms.Path(process.prod1 + process.prod2 + process.test1)

process.endpath = cms.EndPath(process.out)
