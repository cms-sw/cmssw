# Default configuration for the InitRootHandlers service

import FWCore.ParameterSet.Config as cms

InitRootHandlers = cms.Service("InitRootHandlers",
    UnloadRootSigHandler = cms.untracked.bool(False),
    ResetRootErrHandler = cms.untracked.bool(True),
    AutoLibraryLoader = cms.untracked.bool(True)
)
