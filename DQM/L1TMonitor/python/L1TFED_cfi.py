import FWCore.ParameterSet.Config as cms

l1tfed = cms.EDFilter("L1TFED",
    disableROOToutput = cms.untracked.bool(True),
    verbose = cms.untracked.bool(False),
    DQMStore = cms.untracked.bool(True)
)


