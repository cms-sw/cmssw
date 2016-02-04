import FWCore.ParameterSet.Config as cms

l1thcaltpg = cms.EDFilter("L1THCALTPG",
    DQMStore = cms.untracked.bool(True),
    disableROOToutput = cms.untracked.bool(True),
    hcaltpgSource = cms.InputTag("hcalDigis","","DQM"),
    verbose = cms.untracked.bool(False)
)


