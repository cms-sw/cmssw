import FWCore.ParameterSet.Config as cms

l1tfed = cms.EDFilter("L1TFED",
    disableROOToutput = cms.untracked.bool(True),
    verbose = cms.untracked.bool(False),
    DQMStore = cms.untracked.bool(True),
    L1FEDS = cms.vint32(745,760,780,812,813),
    rawTag = cms.InputTag( "source" )
)


