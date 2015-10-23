import FWCore.ParameterSet.Config as cms

from CondCore.CondDB.CondDB_cfi import *

GlobalTag = cms.ESSource("PoolDBESSource",
    CondDB,
    globaltag = cms.string(''),
    snapshotTime = cms.string(''),
    RefreshAlways    = cms.untracked.bool(False),
    RefreshOpenIOVs  = cms.untracked.bool(False),
    RefreshEachRun   = cms.untracked.bool(False),
    ReconnectEachRun = cms.untracked.bool(False),
    DumpStat=cms.untracked.bool(False),
    toGet = cms.VPSet( )   # hook to override or add single payloads
)
