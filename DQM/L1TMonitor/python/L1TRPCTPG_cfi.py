import FWCore.ParameterSet.Config as cms

l1trpctpg = cms.EDAnalyzer("L1TRPCTPG",
    disableROOToutput = cms.untracked.bool(True),
    rpctpgSource = cms.InputTag("rpcunpacker"),
    rpctfSource = cms.InputTag("l1GtUnpack"),
    verbose = cms.untracked.bool(False),
    DQMStore = cms.untracked.bool(True)
)


