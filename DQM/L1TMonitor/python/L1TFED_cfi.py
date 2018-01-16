import FWCore.ParameterSet.Config as cms

l1tfed = DQMStep1Module('L1TFED',
    verbose = cms.untracked.bool(False),
    L1FEDS = cms.vint32(745,760,780,812,813),
    rawTag = cms.InputTag( "source" ),
    FEDDirName = cms.untracked.string("L1T/FEDIntegrity"),
    stableROConfig = cms.untracked.bool(False)
)


