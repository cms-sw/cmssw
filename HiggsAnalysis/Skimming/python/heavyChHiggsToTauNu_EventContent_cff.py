import FWCore.ParameterSet.Config as cms

# Can insert block to customize what goes inside root file on top of AOD/RECO
heavyChHiggsToTauNuEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
heavyChHiggsToTauNuEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('heavyChHiggsToTauNuFilterPath')
    )
)

