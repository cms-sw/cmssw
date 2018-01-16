import FWCore.ParameterSet.Config as cms

l1tcompare = DQMStep1Module('L1TCompare',
    ecalTpgSource = cms.InputTag("ecalTriggerPrimitiveDigis"),
    verbose = cms.untracked.bool(True),
    DQMStore = cms.untracked.bool(True),
    disableROOToutput = cms.untracked.bool(True),
    gctSource = cms.InputTag("gctDigis"),
    rctSource = cms.InputTag("gctDigis")
)


