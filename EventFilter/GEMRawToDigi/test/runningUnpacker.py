import FWCore.ParameterSet.Config as cms

process = cms.Process("RAW")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'WARNING'

process.load('EventFilter.GEMRawToDigi.gemRawToDigi_cfi')
process.load('EventFilter.GEMRawToDigi.GEMSQLiteCabling_cfi')

process.source = cms.Source("EmptySource",   
      firstEvent = cms.untracked.uint32(1),
      firstRun = cms.untracked.uint32(1)
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10))

process.rawDataCollector = cms.EDProducer('GEMDatToRaw',
      inputFileName=cms.string("run304140_ls0001_streamA_StorageManager.dat"),
      ##verbose=cms.untracked.bool(True)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('out.root')
)

#process.check = cms.EDAnalyzer("GEMDigiReader",
#      InputLabel = cms.InputTag("gemDigis"))
  
process.p = cms.Path(process.rawDataCollector
                         #+process.gemRawToDigi
)

process.e = cms.EndPath(process.out)
