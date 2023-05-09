import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.source = cms.Source("EmptySource")
process.maxEvents.input = 1

process.fedRawDataCollectionProducer = cms.EDProducer("TestWriteFEDRawDataCollection",
    # Test values below are meaningless. We just make sure when we read
    # we get the same values.
    FEDData0 = cms.vuint32(0, 1, 2, 3, 4, 5, 6, 7),
    FEDData3 = cms.vuint32(100, 101, 102, 103, 104, 105, 106, 107)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testFEDRawDataCollection.root')
)

process.path = cms.Path(process.fedRawDataCollectionProducer)
process.endPath = cms.EndPath(process.out)
