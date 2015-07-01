import FWCore.ParameterSet.Config as cms

l1tRct = cms.EDAnalyzer("L1TRCT",
    DQMStore = cms.untracked.bool(True),
    disableROOToutput = cms.untracked.bool(True),
    rctSource = cms.InputTag("l1RctHwDigis"),
    gctSource = cms.InputTag("gctDigis"),
    verbose = cms.untracked.bool(False),
    filterTriggerType = cms.int32(1),
    selectBX=cms.untracked.int32(2)
)


