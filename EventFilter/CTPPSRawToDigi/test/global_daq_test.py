import FWCore.ParameterSet.Config as cms

process = cms.Process("TotemIntegratedRawDataTest")

# minimum of logs
process.MessageLogger = cms.Service("MessageLogger",
    statistics = cms.untracked.vstring(),
    destinations = cms.untracked.vstring('cerr'),
    cerr = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    )
)

# raw data source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
      "file:/afs/cern.ch/user/j/jkaspar/public/run268608_ls0001_streamA_StorageManager.root",
      "file:/afs/cern.ch/user/j/jkaspar/public/run281994.root"
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# raw-to-digi conversion
process.load('EventFilter.CTPPSRawToDigi.totemRawToDigi_cff')

# execution configuration
process.p = cms.Path(
    process.totemTriggerRawToDigi *
    process.totemRPRawToDigi
)
