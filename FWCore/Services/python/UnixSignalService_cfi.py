# Default configuration for the UnixSignalService

import FWCore.ParameterSet.Config as cms

UnixSignalService = cms.Service("UnixSignalService",
    EnableCtrlC = cms.untracked.bool(True)
)
