import FWCore.ParameterSet.Config as cms

l1thcaltpgxana = cms.EDFilter("L1THCALTPGXAna",
    hfrecoSource = cms.InputTag("hfreco","","DQM"),
    fakeCut = cms.untracked.double(5.0),
    outputFile = cms.untracked.string('./L1TDQM.root'),
    MonitorDaemon = cms.untracked.bool(True),
    hcaltpgSource = cms.InputTag("hcalDigis","","DQM"),
    hbherecoSource = cms.InputTag("hbhereco","","DQM"),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    verbose = cms.untracked.bool(False)
)


