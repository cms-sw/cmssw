
import FWCore.ParameterSet.Config as cms

# Can insert block to customize what goes inside root file on top of AOD/RECO
lightChHiggsToTauNuEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
lightChHiggsToTauNuEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('lightChHiggsToTauNuFilterPath')
    )
)

