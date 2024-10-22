import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.source = cms.Source("EmptySource")
process.maxEvents.input = 10

process.collectionProducer = cms.EDProducer("TestWriteVectorDetId",
    # Test values below are meaningless. We just make sure when we read
    # we get the same values.
    testValue = cms.uint32(21)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testVectorDetId.root')
)

process.path = cms.Path(process.collectionProducer)
process.endPath = cms.EndPath(process.out)
