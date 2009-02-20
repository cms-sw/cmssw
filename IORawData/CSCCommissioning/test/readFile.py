import FWCore.ParameterSet.Config as cms

process = cms.Process("reader")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.placeholder = cms.untracked.bool(False)
process.MessageLogger.cout.threshold = cms.untracked.string('INFO')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

process.source = cms.Source("DaqSource",
  readerPluginName = cms.untracked.string("CSCFileReader"),
  readerPset = cms.untracked.PSet(
     tfDDUnumber = cms.uint32(0),
     firstEvent  = cms.uint32(0),
     RUI01  = cms.untracked.vstring('/tmp/kkotov/66637_.bin_760'),
     FED760 = cms.untracked.vstring('RUI01')
  )
)


