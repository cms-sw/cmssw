import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("DaqSource",
    readerPluginName = cms.untracked.string('DaqFakeReader'),
    readerPset = cms.untracked.PSet(

    )
)

process.dummyunpacker = cms.EDAnalyzer("DummyUnpackingModule", fedRawDataCollectionTag = cms.InputTag('rawDataCollector'))

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('.rawdata.root')
)

process.ep = cms.EndPath(process.out)
process.p = cms.Path(process.dummyunpacker)


