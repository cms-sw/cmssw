import FWCore.ParameterSet.Config as cms

from CondCore.CondDB.CondDB_cfi import *

GlobalTag = cms.ESSource("PoolDBESSource",
    CondDBSetup,
    RefreshAlways    = cms.untracked.bool(False),
    RefreshOpenIOVs  = cms.untracked.bool(False),
    RefreshEachRun   = cms.untracked.bool(False),
    ReconnectEachRun = cms.untracked.bool(False),
    snapshotTime = cms.string(''),
    connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'),
    globaltag = cms.string(''),
    toGet = cms.VPSet( )   # hook to override or add single payloads
)
