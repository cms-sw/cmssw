import FWCore.ParameterSet.Config as cms

l1demonecal = cms.EDFilter("L1TdeECAL",
    VerboseFlag = cms.untracked.int32(0),
    DataEmulCompareSource = cms.InputTag("l1compare"),
    MonitorDaemon = cms.untracked.bool(True),
    HistFile = cms.untracked.string('l1demon.root'),
    DaqMonitorBEInterface = cms.untracked.bool(True)
)


