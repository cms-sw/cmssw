import FWCore.ParameterSet.Config as cms

# needed backend
DQMStore = cms.Service("DQMStore",
    referenceFileName = cms.untracked.string(''),
    verbose = cms.untracked.int32(0),
    collateHistograms = cms.bool(False)
)

# needed output
from DQMServices.Components.DQMEnvironment_cfi import *

# actual producer
from DQMServices.Components.EDMtoMEConverter_cfi import *

