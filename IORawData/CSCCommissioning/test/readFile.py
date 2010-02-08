import FWCore.ParameterSet.Config as cms

process = cms.Process("reader")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.placeholder = cms.untracked.bool(False)
process.MessageLogger.cout.threshold = cms.untracked.string('INFO')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

process.source = cms.Source("DaqSource",
    readerPluginName = cms.untracked.string('CSCFileReader'),
    readerPset = cms.untracked.PSet(
        firstEvent  = cms.untracked.int32(0),
        tfDDUnumber = cms.untracked.int32(0),
        FED760 = cms.untracked.vstring('RUI00'),
        RUI00  = cms.untracked.vstring('/tmp/kkotov/qwe.raw')
  )
)


