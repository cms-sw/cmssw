import FWCore.ParameterSet.Config as cms

process = cms.Process("CTPPSRawToDigiTestDiamondsOnly")

# minimum of logs
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG')
    )
)

# raw data source
process.source = cms.Source("NewEventStreamFileReader",
  fileNames = cms.untracked.vstring(
        '/store/t0streamer/Minidaq/A/000/298/442/run298442_ls0001_streamA_StorageManager.dat'
  )
)

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(20)
)

# raw-to-digi conversion
process.load("EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff")
 
process.p = cms.Path(
  process.ctppsDiamondRawToDigi
)

# output configuration
process.output = cms.OutputModule("PoolOutputModule",
  fileName = cms.untracked.string("file:./reco_diamond_digi.root"),
  outputCommands = cms.untracked.vstring(
    'drop *',
    'keep *_ctppsDiamondRawToDigi_*_*',
 )
)

process.outpath = cms.EndPath(process.output)
