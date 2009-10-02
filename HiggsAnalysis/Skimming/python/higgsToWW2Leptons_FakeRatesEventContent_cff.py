import FWCore.ParameterSet.Config as cms

# Can insert block to customize what goes inside root file on top of AOD/RECO
higgsToWW2LeptonsFakeRatesEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
higgsToWW2LeptonsFakeRatesEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('HWWFakeRatesFilterPath')
    )
)

