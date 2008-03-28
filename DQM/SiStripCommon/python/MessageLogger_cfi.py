# The following comments couldn't be translated into the new config version:

#@@ comment to suppress debug statements!

# allows to suppress output from specific modules 

import FWCore.ParameterSet.Config as cms

MessageLogger = cms.Service("MessageLogger",
    info = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
        limit = cms.untracked.int32(100000),
        noLineBreaks = cms.untracked.bool(False)
    ),
    suppressInfo = cms.untracked.vstring(),
    suppressDebug = cms.untracked.vstring(),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING'),
        limit = cms.untracked.int32(100000),
        noLineBreaks = cms.untracked.bool(False)
    ),
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
    debugModules = cms.untracked.vstring('*'),
    destinations = cms.untracked.vstring('cout', 'cerr', 'debug', 'info', 'warning', 'error')
)


