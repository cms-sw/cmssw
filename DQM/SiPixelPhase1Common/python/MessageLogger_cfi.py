import FWCore.ParameterSet.Config as cms

MessageLogger = cms.Service(
    "MessageLogger",
    
    debug = cms.untracked.PSet(
    threshold = cms.untracked.string('DEBUG'),
    limit = cms.untracked.uint32(100000),
    noLineBreaks = cms.untracked.bool(True),
    ),

    cerr = cms.untracked.PSet(
    threshold = cms.untracked.string('WARNING'),
    limit = cms.untracked.uint32(100000),
    noLineBreaks = cms.untracked.bool(False),
    ),

    destinations = cms.untracked.vstring(
    'debug', 
    'cerr'
    ),
    
    #@@ comment to suppress debug statements!
    debugModules = cms.untracked.vstring('*'),
    
    # allows to suppress output from specific modules 
    suppressDebug = cms.untracked.vstring(),
    suppressInfo = cms.untracked.vstring(),
    suppressWarning = cms.untracked.vstring(),
    
    )


