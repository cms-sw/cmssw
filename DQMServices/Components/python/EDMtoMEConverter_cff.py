# The following comments couldn't be translated into the new config version:

# needed backend

import FWCore.ParameterSet.Config as cms

# needed output
from DQMServices.Components.test.dqm_onlineEnv_cfi import *
# actual producer
from DQMServices.Components.EDMtoMEConverter_cfi import *
DQMStore = cms.Service("DQMStore",
    referenceFileName = cms.untracked.string(''),
    verbose = cms.untracked.int32(0),
    collateHistograms = cms.untracked.bool(True)
)


