import FWCore.ParameterSet.Config as cms

rpcdigidqm = cms.EDFilter("RPCMonitorDigi",
    moduleLogName = cms.untracked.string('DigiModule'),
    DigiDQMSaveRootFile = cms.untracked.bool(False),
    DigiEventsInterval = cms.untracked.int32(500),
    dqmsuperexpert = cms.untracked.bool(False),
    dqmexpert = cms.untracked.bool(False),
    dqmshifter = cms.untracked.bool(False),
    RootFileNameDigi = cms.untracked.string('RPCMonitorDigi.root')
)


