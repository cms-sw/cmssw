import FWCore.ParameterSet.Config as cms

# FastTimerService
from HLTrigger.Timer.FastTimerService_cfi import *

# DQM file saver
dqmFileSaver = cms.EDAnalyzer( "DQMFileSaver",
  convention        = cms.untracked.string( "Offline" ),
  workflow          = cms.untracked.string( "/HLT/FastTimerService/All" ),
  dirName           = cms.untracked.string( "." ),
  saveByRun         = cms.untracked.int32(  1 ),
  saveByLumiSection = cms.untracked.int32( -1 ),
  saveAtJobEnd      = cms.untracked.bool( False ),
 #forceRunNumber    = cms.untracked.int32( 999999 )
)

DQMFileSaverOutput = cms.EndPath( dqmFileSaver )
