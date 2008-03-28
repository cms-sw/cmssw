import FWCore.ParameterSet.Config as cms

MonitorDaemon = cms.Service("MonitorDaemon",
    AutoInstantiate = cms.untracked.bool(True),
    DestinationAddress = cms.untracked.string('localhost'),
    SendPort = cms.untracked.int32(9090),
    NameAsSource = cms.untracked.string('SiStripCommissioningSource'),
    UpdateDelay = cms.untracked.int32(100),
    reconnect_delay = cms.untracked.int32(5)
)


