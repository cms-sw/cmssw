import FWCore.ParameterSet.Config as cms

# Can insert block to customize what goes inside root file on top of AOD/RECO
QCDHLT1jetPE1EventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
QCDHLT1jetPE1EventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('QCDHLT1jetPE1Skimpath')
    )
)

