import FWCore.ParameterSet.Config as cms

MLlog4cplus = cms.Service("MLlog4cplus")
#MessageLogger = cms.Service("MessageLogger",
#    suppressWarning = cms.untracked.vstring(),
#    log4cplus = cms.untracked.PSet(
#        threshold = cms.untracked.string('DEBUG')
#    ),
#    suppressDebug = cms.untracked.vstring(),
#    debugModules = cms.untracked.vstring(),
#    suppressInfo = cms.untracked.vstring()
#)
