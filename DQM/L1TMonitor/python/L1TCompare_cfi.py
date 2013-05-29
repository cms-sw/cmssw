import FWCore.ParameterSet.Config as cms

l1tcompare = cms.EDAnalyzer("L1TCompare",
    ecalTpgSource = cms.InputTag("ecalTriggerPrimitiveDigis"),
    verbose = cms.untracked.bool(True),
    DQMStore = cms.untracked.bool(True),
    disableROOToutput = cms.untracked.bool(True),
    gctSource = cms.InputTag("l1GctHwDigis","","DQM"),
    rctSource = cms.InputTag("l1GctHwDigis","","DQM")
)


