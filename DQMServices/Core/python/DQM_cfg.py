import FWCore.ParameterSet.Config as cms

# needed backend
from DQMServices.Core.DQMStore_cfg import *

DQM = cms.Service("DQM",
    debug = cms.untracked.bool(False),
    publishFrequency = cms.untracked.double(5.0),
    collectorPort = cms.untracked.int32(9090),
    collectorHost = cms.untracked.string('localhost'),
    filter = cms.untracked.string('')
)
