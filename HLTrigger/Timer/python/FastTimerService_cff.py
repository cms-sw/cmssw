import FWCore.ParameterSet.Config as cms

# FastTimerService
from HLTrigger.Timer.FastTimerService_cfi import *
from HLTrigger.Timer.fastTimerServiceClient_cfi import *

# DQM file saver
dqmFileSaver = cms.EDAnalyzer( "DQMFileSaver",
#   producer          = cms.untracked.string('DQM'),
#   version           = cms.untracked.int32(1),
#   referenceRequireStatus = cms.untracked.int32(100),
#   runIsComplete     = cms.untracked.bool(False),
#   referenceHandling = cms.untracked.string('all'),
    convention        = cms.untracked.string( "Offline" ),
    workflow          = cms.untracked.string( "/HLT/FastTimerService/All" ),
    dirName           = cms.untracked.string( "." ),
    saveByRun         = cms.untracked.int32(1),
    saveByLumiSection = cms.untracked.int32(-1),
    saveByEvent       = cms.untracked.int32(-1),
    saveByTime        = cms.untracked.int32(-1),
    saveByMinute      = cms.untracked.int32(-1),
    saveAtJobEnd      = cms.untracked.bool(False),
    forceRunNumber    = cms.untracked.int32(-1),
)

DQMFileSaverOutput = cms.EndPath( fastTimerServiceClient + dqmFileSaver )
