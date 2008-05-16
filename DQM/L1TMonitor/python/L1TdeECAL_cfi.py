import FWCore.ParameterSet.Config as cms

l1demonecal = cms.EDFilter("L1TdeECAL",
    HistFolder = cms.untracked.string('L1TEMU/xpert/Ecal/'),
    verbose = cms.untracked.bool(False),
    HistFile = cms.untracked.string('./L1TDQM.root'),
    DataEmulCompareSource = cms.InputTag("l1compare"),
    DQMStore = cms.untracked.bool(True),
    disableROOToutput = cms.untracked.bool(True)
)


