import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.source = cms.Source("EmptySource")
process.maxEvents.input = 10

process.collectionProducer = cms.EDProducer("TestWriteSiStripApproximateClusterCollection",
    # Test values below are meaningless. We just make sure when we read
    # we get the same values.
    integralValues = cms.vuint32(11, 21, 31, 41, 51, 62, 73)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testSiStripApproximateClusterCollection.root')
)

process.path = cms.Path(process.collectionProducer)
process.endPath = cms.EndPath(process.out)
