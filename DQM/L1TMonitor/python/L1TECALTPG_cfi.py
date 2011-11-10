import FWCore.ParameterSet.Config as cms

l1tecaltpg = cms.EDAnalyzer("L1TECALTPG",
    ecaltpgSourceE = cms.InputTag("ecalEBunpacker","EcalTriggerPrimitives"),
    disableROOToutput = cms.untracked.bool(True),
    ecaltpgSourceB = cms.InputTag("ecalEBunpacker","EcalTriggerPrimitives"),
    verbose = cms.untracked.bool(False),
    DQMStore = cms.untracked.bool(True)
)


