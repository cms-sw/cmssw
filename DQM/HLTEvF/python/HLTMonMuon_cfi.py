import FWCore.ParameterSet.Config as cms

hltMuonMonitor = cms.EDFilter("HLTMuonDQMSource",
    MonitorDaemon = cms.untracked.bool(True),
    verbose = cms.untracked.bool(False),
    DaqMonitorBEInterface = cms.untracked.bool(True)
)

hltMonMuonReco = cms.Sequence(hltMuonMonitor)

