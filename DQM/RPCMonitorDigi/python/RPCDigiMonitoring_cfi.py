import FWCore.ParameterSet.Config as cms

rpcdigidqm = cms.EDFilter("RPCMonitorDigi",
    moduleLogName = cms.untracked.string('DigiModule'),
    dqmexpert = cms.untracked.bool(False),
    DigiEventsInterval = cms.untracked.int32(500),
    MergeDifferentRuns = cms.untracked.bool(False),
    dqmsuperexpert = cms.untracked.bool(False),
    DigiDQMSaveRootFile = cms.untracked.bool(False),
    dqmshifter = cms.untracked.bool(False),
    RootFileNameDigi = cms.untracked.string('RPCMonitorDigi.root')
)


