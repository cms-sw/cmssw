import FWCore.ParameterSet.Config as cms

l1tfed = cms.EDAnalyzer("L1TFED",
    disableROOToutput = cms.untracked.bool(False),
    verbose = cms.untracked.bool(True),
    DQMStore = cms.untracked.bool(True),
    L1FEDS = cms.vint32(745,760,780,812,813),
    rawTag = cms.InputTag( "source" ),
    FEDDirName = cms.untracked.string("L1T/FEDIntegrity"),
    stableROConfig = cms.untracked.bool(False)
)


