import FWCore.ParameterSet.Config as cms

# Can insert block to customize what goes inside root file on top of AOD/RECOSIM
higgsToInvisibleEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
higgsToInvisibleEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('HToInvisFilterPath')
    )
)

