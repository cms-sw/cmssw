import FWCore.ParameterSet.Config as cms

# Can insert block to customize what goes inside root file on top of AOD/RECO
QCDHLT1jetEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
QCDHLT1jetEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('QCDHLT1jetSkimpath')
    )
)

