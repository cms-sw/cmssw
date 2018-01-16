import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
l1tcompare = DQMEDAnalyzer('L1TCompare',
    ecalTpgSource = cms.InputTag("ecalTriggerPrimitiveDigis"),
    verbose = cms.untracked.bool(True),
    DQMStore = cms.untracked.bool(True),
    disableROOToutput = cms.untracked.bool(True),
    gctSource = cms.InputTag("gctDigis"),
    rctSource = cms.InputTag("gctDigis")
)


