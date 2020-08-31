import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST2")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testParentageWithStreamerIO1.root'
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.prod11 = cms.EDProducer("AddIntsProducer",
    labels = cms.VInputTag()
)

process.prod12 = cms.EDProducer("AddIntsProducer",
    labels = cms.VInputTag("prod11")
)

process.test1 = cms.EDAnalyzer("TestParentage",
    inputTag = cms.InputTag("prod2"),
    expectedAncestors = cms.vstring()
)

process.test2 = cms.EDAnalyzer("TestParentage",
    inputTag = cms.InputTag("prod12"),
    expectedAncestors = cms.vstring("prod11")
)

process.out = cms.OutputModule("EventStreamFileWriter",
    fileName = cms.untracked.string('testParentageWithStreamerIO2.dat'),
    compression_level = cms.untracked.int32(1),
    use_compression = cms.untracked.bool(True),
    max_event_size = cms.untracked.int32(7000000)
)

process.path1 = cms.Path(process.prod11 + process.prod12 + process.test1 + process.test2)

process.endpath = cms.EndPath(process.out)
