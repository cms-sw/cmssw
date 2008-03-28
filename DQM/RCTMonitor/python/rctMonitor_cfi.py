import FWCore.ParameterSet.Config as cms

rctmonitor = cms.EDFilter("RCTMonitor",
    OutputFileName = cms.untracked.string('rctMonitor.root'),
    #untracked InputTag rctSource  = l1RctEmulDigis // for MC file
    WriteOutputFile = cms.untracked.bool(True),
    EnableRctHistos = cms.untracked.bool(True),
    EnableMonitorDaemon = cms.untracked.bool(False),
    # Labels for RCT digis
    rctSource = cms.untracked.InputTag("l1GctHwDigis")
)


