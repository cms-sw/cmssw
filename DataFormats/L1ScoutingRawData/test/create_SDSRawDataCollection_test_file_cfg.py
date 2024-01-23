import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.source = cms.Source("EmptySource")
process.maxEvents.input = 1

process.sdsRawDataCollectionProducer = cms.EDProducer("TestWriteSDSRawDataCollection",
    # Test values below are meaningless. We just make sure when we read
    # we get the same values.
    SDSData1 = cms.vuint32(0, 1, 2, 3),
    SDSData2 = cms.vuint32(42, 43, 44, 45)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testSDSRawDataCollection.root')
)

process.path = cms.Path(process.sdsRawDataCollectionProducer)
process.endPath = cms.EndPath(process.out)