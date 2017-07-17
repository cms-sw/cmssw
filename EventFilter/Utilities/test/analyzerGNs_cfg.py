import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1),
    output = cms.untracked.int32(-1)
)

process.options = cms.untracked.PSet(
    SkipEvent = cms.untracked.vstring('ProductNotFound')
    )
process.source = cms.Source("FRDStreamSource",
    fileNames = cms.untracked.vstring(
'file:/home/meschi/CMSSW_7_1_X_2014-09-12-0200/src/EventFilter/FEDInterface/data/fed1024data/run226295_ls0302_index000000.raw'    )
)
process.a = cms.EDAnalyzer("GlobalNumbersAnalysis");

process.b = cms.EDAnalyzer("DumpFEDRawDataProduct",
			   label = cms.untracked.string("source"),
                           feds = cms.untracked.vint32(1024),
                           dumpPayload = cms.untracked.bool(True)
                           )

# path to be run
process.p = cms.Path(process.a+process.b)

#process.ep = cms.EndPath(process.out)


