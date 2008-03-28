import FWCore.ParameterSet.Config as cms

l1demon = cms.EDFilter("L1TDEMON",
    VerboseFlag = cms.untracked.int32(0),
    HistFile = cms.untracked.string('l1demon.root'),
    MonitorDaemon = cms.untracked.bool(True),
    DataEmulCompareSource = cms.InputTag("l1compare"),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    disableROOToutput = cms.untracked.bool(True)
)


