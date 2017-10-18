import FWCore.ParameterSet.Config as cms

process = cms.Process("RAW")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'WARNING'

process.load("EventFilter.GEMRawToDigi.GEMSQLiteCabling_cfi")
process.GEMCabling.connect = 'sqlite_file:GEMEMap_CosmicStand_8Nov2016.db'

SLOTLIST=[]
VFATLIST=[]
COLUMNLIST=[]
ROWLIST=[]
LAYERLIST=[]

import configureRun_cfi
configureRun_cfi.configureRun(SLOTLIST,VFATLIST,COLUMNLIST,ROWLIST,LAYERLIST)

process.source = cms.Source("EmptySource",   
      firstEvent = cms.untracked.uint32(1),
      firstRun = cms.untracked.uint32(configureRun_cfi.RunNumber)
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(configureRun_cfi.MaxEvents))

process.gemDigis = cms.EDProducer('GEMCosmicStandUnpacker',
      inputFileName=cms.string(configureRun_cfi.RAWFileName),
      slotVector=  cms.vint32(SLOTLIST),
      vfatVector=cms.vuint64(VFATLIST),
      columnVector=cms.vint32(COLUMNLIST),
      rowVector   =cms.vint32(ROWLIST),
      layerVector = cms.vint32(LAYERLIST),
      ##verbose=cms.untracked.bool(True)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string(configureRun_cfi.OutputFileName)
)

process.check = cms.EDAnalyzer("GEMDigiReader",
      InputLabel = cms.InputTag("gemDigis"))
  
process.p = cms.Path(process.gemDigis)

process.e = cms.EndPath(process.out)
