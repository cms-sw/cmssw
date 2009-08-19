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
process.MessageLogger = cms.Service(
    "MessageLogger",
    info = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
        #limit = cms.untracked.int32(100000),
        noLineBreaks = cms.untracked.bool(False)
        ),
    suppressInfo = cms.untracked.vstring(),
    # allows to suppress output from specific modules 
    suppressDebug = cms.untracked.vstring(),
    debug = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG'),
        #limit = cms.untracked.int32(100000),
        noLineBreaks = cms.untracked.bool(False)
        ),
    warning = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING'),
        #limit = cms.untracked.int32(100000),
        noLineBreaks = cms.untracked.bool(False)
        ),
    cerr = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR'),
        #limit = cms.untracked.int32(100000),
        noLineBreaks = cms.untracked.bool(False)
        ),
    error = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR'),
        #limit = cms.untracked.int32(100000),
        noLineBreaks = cms.untracked.bool(False)
        ),
    suppressWarning = cms.untracked.vstring(),
    #debugModules = cms.untracked.vstring('*'),#'siStripFEDMonitor'),
    destinations = cms.untracked.vstring('cerr', 
                                         'debug', 
                                         'info', 
                                         'warning', 
                                         'error')

)

process.load('DQM.SiStripMonitorHardware.siStripFEDDump_cfi')
process.siStripFEDDump.FEDID = 260

process.p = cms.Path( process.siStripFEDDump )
