import FWCore.ParameterSet.Config as cms
from DQMServices.Components.EDMtoMEConverter_cfi import *

DaqMonitorROOTBackEnd = cms.Service("DaqMonitorROOTBackEnd",
    referenceFileName = cms.untracked.string(''),
    verbose = cms.untracked.int32(0),
    collateHistograms = cms.untracked.bool(True)
)
