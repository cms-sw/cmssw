import FWCore.ParameterSet.Config as cms

l1tdttpg = DQMStep1Module('L1TDTTPG',
    disableROOToutput = cms.untracked.bool(True),
    dttpgSource = cms.InputTag("muonDTDigis"),
    verbose = cms.untracked.bool(False),
    DQMStore = cms.untracked.bool(True)
)


