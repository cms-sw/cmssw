import FWCore.ParameterSet.Config as cms

# Can insert block to customize what goes inside root file on top of AOD/RECO
higgsToZZ4LeptonsEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
higgsToZZ4LeptonsEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('HZZFilterPath')
    )
)

