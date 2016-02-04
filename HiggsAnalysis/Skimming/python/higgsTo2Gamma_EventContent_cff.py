import FWCore.ParameterSet.Config as cms

# Can insert block to customize what goes inside root file on top of AOD/RECO
higgsTo2GammaEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
higgsTo2GammaEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('HggFilterPath')
    )
)

