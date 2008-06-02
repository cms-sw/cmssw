import FWCore.ParameterSet.Config as cms

MessageLogger = cms.Service("MessageLogger",
    info = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
        limit = cms.untracked.int32(100000),
        noLineBreaks = cms.untracked.bool(False)
    ),
    suppressInfo = cms.untracked.vstring(),
    # allows to suppress output from specific modules 
    suppressDebug = cms.untracked.vstring(),
    debug = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG'),
        limit = cms.untracked.int32(100000),
        noLineBreaks = cms.untracked.bool(False)
    ),
    warning = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING'),
        limit = cms.untracked.int32(100000),
        noLineBreaks = cms.untracked.bool(False)
    ),
    cerr = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR'),
        limit = cms.untracked.int32(100000),
        noLineBreaks = cms.untracked.bool(False)
    ),
    error = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR'),
        limit = cms.untracked.int32(100000),
        noLineBreaks = cms.untracked.bool(False)
    ),
    suppressWarning = cms.untracked.vstring(),
    #@@ comment to suppress debug statements!
    debugModules = cms.untracked.vstring('*'),
    destinations = cms.untracked.vstring('cerr', 
        'debug', 
        'info', 
        'warning', 
        'error')
)


