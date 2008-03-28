# The following comments couldn't be translated into the new config version:

# needed backend

import FWCore.ParameterSet.Config as cms
# actual producer
from DQMServices.Components.EDMtoMEConverter_cfi import *

DQMStore = cms.Service("DQMStore",
    # default ""
    referenceFileName = cms.untracked.string(''),
    # default 0
    verbose = cms.untracked.int32(0),
    # default true
    collateHistograms = cms.untracked.bool(True)
)
