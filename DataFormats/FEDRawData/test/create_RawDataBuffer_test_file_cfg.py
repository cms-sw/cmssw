import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.source = cms.Source("EmptySource")
process.maxEvents.input = 1

process.rawDataBufferProducer = cms.EDProducer("TestWriteRawDataBuffer",
    # Test values below are meaningless. We just make sure when we read
    # we get the same values.
    dataPattern1 = cms.vuint32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
    dataPattern2 = cms.vuint32(100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testRawDataBuffer.root')
)

process.path = cms.Path(process.rawDataBufferProducer)
process.endPath = cms.EndPath(process.out)
