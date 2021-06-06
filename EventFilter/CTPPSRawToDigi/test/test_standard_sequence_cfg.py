import FWCore.ParameterSet.Config as cms

process = cms.Process("CTPPSRawToDigiTestStandardSequence")

# minimum of logs
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    )
)

# raw data source
process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
#    '/store/express/Run2016H/ExpressPhysics/FEVT/Express-v2/000/283/877/00000/4EE44B0E-2499-E611-A155-02163E011938.root'
    '/store/data/Run2016H/ZeroBias/RAW/v1/000/283/820/00000/462D2A5B-B19A-E611-B100-02163E01382E.root'
  )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

# raw-to-digi conversion
process.load("EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff")

# execution configuration
process.p = cms.Path(
    process.ctppsRawToDigi
)

# output configuration
process.output = cms.OutputModule("PoolOutputModule",
  fileName = cms.untracked.string("file:./reco_digi.root"),
  outputCommands = cms.untracked.vstring(
    'drop *',
    'keep *_*RawToDigi_*_*',
  )
)

process.outpath = cms.EndPath(process.output)
