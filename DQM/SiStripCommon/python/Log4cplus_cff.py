import FWCore.ParameterSet.Config as cms

MLlog4cplus = cms.Service("MLlog4cplus")

MessageLogger = cms.Service("MessageLogger",
    suppressWarning = cms.untracked.vstring(),
    # Threshold for messages streamed to log4cplus
    log4cplus = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG')
    ),
    # allows to suppress output from specific modules 
    suppressDebug = cms.untracked.vstring(),
    #@@ comment to suppress debug statements!
    debugModules = cms.untracked.vstring('*'),
    suppressInfo = cms.untracked.vstring()
)


