import FWCore.ParameterSet.Config as cms
from DQMServices.Components.EDMtoMEConverter_cfi import *

DQMStore = cms.Service("DQMStore",
    referenceFileName = cms.untracked.string(''),
    verbose = cms.untracked.int32(0),
    collateHistograms = cms.untracked.bool(True)
)
