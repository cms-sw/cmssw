import FWCore.ParameterSet.Config as cms

rpcsyncdqm = cms.EDFilter("RPCMonitorSync",
    SyncEventsInterval = cms.untracked.int32(100),
    RootFileNameSync = cms.untracked.string('RPCMonitorSync.root'),
    moduleLogName = cms.untracked.string('SyncModule'),
    SyncDQMSaveRootFile = cms.untracked.bool(False)
)


