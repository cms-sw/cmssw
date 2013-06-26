import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1),
    output = cms.untracked.int32(-1)
)

process.options = cms.untracked.PSet(
    SkipEvent = cms.untracked.vstring('ProductNotFound')
    )
process.source = cms.Source("NewEventStreamFileReader",
    fileNames = cms.untracked.vstring(
'file:/store/global/00/closed/Data.00134542.0001.A.storageManager.08.0000.dat'
    )
)
process.a = cms.EDAnalyzer("GlobalNumbersAnalysis");

process.b = cms.EDAnalyzer("DumpFEDRawDataProduct",
			   label = cms.untracked.string("source"),
                           feds = cms.untracked.vint32(812,1023),
                           dumpPayload = cms.untracked.bool(False)
                           )
process.c = cms.EDAnalyzer("EvFRecordUnpacker",
                           inputTag = cms.InputTag("source")
                           )
process.out = cms.OutputModule("PoolOutputModule",fileName=cms.untracked.string("file:pippo.root"))

# path to be run
process.p = cms.Path(process.b+process.c)

#process.ep = cms.EndPath(process.out)


