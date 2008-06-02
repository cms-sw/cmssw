import FWCore.ParameterSet.Config as cms

l1demonecal = cms.EDFilter("L1TdeECAL",
    disableROOToutput = cms.untracked.bool(True),
    outputFile = cms.untracked.string('./L1TDQM.root'),
    DataEmulCompareSource = cms.InputTag("l1compare"),
    verbose = cms.untracked.bool(False),
    DQMStore = cms.untracked.bool(True)
)


