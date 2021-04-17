import FWCore.ParameterSet.Config as cms

process = cms.Process("CTPPSRawToDigiTestStandardSequence")

# minimum of logs
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    )
)

# raw data source
process.source = cms.Source("NewEventStreamFileReader",
    fileNames = cms.untracked.vstring(
#        '/store/express/Run2016H/ExpressPhysics/FEVT/Express-v2/000/283/877/00000/4EE44B0E-2499-E611-A155-02163E011938.root'
        'file:/eos/cms/store/group/dpg_ctpps/comm_ctpps/TotemTiming/Minidaq/519/run312519_ls0002_streamA_StorageManager.dat'
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.verbosity = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

# raw-to-digi conversion
process.load("EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff")

# execution configuration
process.p = cms.Path(
    process.ctppsRawToDigi
)

# output configuration
process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string("file:./totemTiming_digi.root"),
    outputCommands = cms.untracked.vstring(
      'drop *',
      'keep *_*totemTimingRawToDigi_*_*',
    )
)

process.outpath = cms.EndPath(process.output)

