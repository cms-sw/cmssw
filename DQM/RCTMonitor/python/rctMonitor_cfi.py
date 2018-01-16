import FWCore.ParameterSet.Config as cms

rctmonitor = DQMStep1Module('RCTMonitor',
    OutputFileName = cms.untracked.string('rctMonitor.root'), ## Name of root file for histograms

    #untracked InputTag rctSource  = l1RctEmulDigis // for MC file
    WriteOutputFile = cms.untracked.bool(True),
    EnableRctHistos = cms.untracked.bool(True), ## Enable RCT histograms

    EnableMonitorDaemon = cms.untracked.bool(False), ## Enable the monitor daemon   

    # Labels for RCT digis
    rctSource = cms.untracked.InputTag("l1GctHwDigis")
)


