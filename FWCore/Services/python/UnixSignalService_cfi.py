# The following comments couldn't be translated into the new config version:

# Default configuration for the UnixSignalService

import FWCore.ParameterSet.Config as cms

UnixSignalService = cms.Service("UnixSignalService",
    EnableCtrlC = cms.untracked.bool(True)
)


