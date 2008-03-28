# The following comments couldn't be translated into the new config version:

#@@ comment to suppress debug statements!

# allows to suppress output from specific modules 

import FWCore.ParameterSet.Config as cms

MLlog4cplus = cms.Service("MLlog4cplus")

MessageLogger = cms.Service("MessageLogger",
    suppressWarning = cms.untracked.vstring(),
    # Threshold for messages streamed to log4cplus
    log4cplus = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG')
    ),
    suppressDebug = cms.untracked.vstring(),
    debugModules = cms.untracked.vstring('*'),
    suppressInfo = cms.untracked.vstring()
)


