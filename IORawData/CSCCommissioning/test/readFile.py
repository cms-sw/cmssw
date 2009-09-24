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
        FED750 = cms.untracked.vstring('RUI01','RUI02','RUI03'),
        RUI00  = cms.untracked.vstring('/tmp/csc_00099909_EmuRUI00_Monitor_000.raw','/tmp/csc_00099909_EmuRUI00_Monitor_001.raw'),
        RUI01  = cms.untracked.vstring('/tmp/csc_00099909_EmuRUI01_Monitor_000.raw'),
        RUI02  = cms.untracked.vstring('/tmp/csc_00099909_EmuRUI02_Monitor_000.raw'),
        RUI03  = cms.untracked.vstring('/tmp/csc_00099909_EmuRUI03_Monitor_000.raw')
  )
)

process.FEVT = cms.OutputModule("PoolOutputModule",
        fileName = cms.untracked.string("/tmp/test.root"),
        outputCommands = cms.untracked.vstring("keep *")
)

process.outpath = cms.EndPath(process.FEVT)
