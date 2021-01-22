import FWCore.ParameterSet.Config as cms

process = cms.Process('TEST')

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
    'file:/home/nc302/cmssw/data/111195/USC.00111195.0001.A.storageManager.00.0000.root'
  )
)

#process.load("DQM.SiStripMonitorHardware.test.source_cff")
process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(1)
)

process.load('FWCore/MessageService/MessageLogger_cfi')
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        noLineBreaks = cms.untracked.bool(False),
        threshold = cms.untracked.string('ERROR')
    ),
    files = cms.untracked.PSet(
        debug = cms.untracked.PSet(
            noLineBreaks = cms.untracked.bool(False),
            threshold = cms.untracked.string('DEBUG')
        ),
        error = cms.untracked.PSet(
            noLineBreaks = cms.untracked.bool(False),
            threshold = cms.untracked.string('ERROR')
        ),
        info = cms.untracked.PSet(
            noLineBreaks = cms.untracked.bool(False),
            threshold = cms.untracked.string('INFO')
        ),
        warning = cms.untracked.PSet(
            noLineBreaks = cms.untracked.bool(False),
            threshold = cms.untracked.string('WARNING')
        )
    ),
    suppressDebug = cms.untracked.vstring(),
    suppressInfo = cms.untracked.vstring(),
    suppressWarning = cms.untracked.vstring()
)

process.load('DQM.SiStripMonitorHardware.siStripFEDDump_cfi')
process.siStripFEDDump.FEDID = 260

process.p = cms.Path( process.siStripFEDDump )
