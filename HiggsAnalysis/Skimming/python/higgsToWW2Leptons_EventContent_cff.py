import FWCore.ParameterSet.Config as cms

# Can insert block to customize what goes inside root file on top of AOD/RECO
higgsToWW2LeptonsEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
higgsToWW2LeptonsEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('HWWFilterPath')
    )
)

