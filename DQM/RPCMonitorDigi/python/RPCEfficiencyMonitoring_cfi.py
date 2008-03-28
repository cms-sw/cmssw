import FWCore.ParameterSet.Config as cms

rpcefficiencydqm = cms.EDFilter("RPCMonitorEfficiency",
    EfficDQMSaveRootFile = cms.untracked.bool(False),
    recHits4DLabel = cms.string('dt4DSegments'),
    EfficEventsInterval = cms.untracked.int32(100),
    RootFileNameEfficiency = cms.untracked.string('RPCMonitorEfficiency.root'),
    moduleLogName = cms.untracked.string('EfficiencyModule')
)


