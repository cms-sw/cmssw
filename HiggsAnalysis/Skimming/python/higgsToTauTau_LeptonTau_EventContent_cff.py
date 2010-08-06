import FWCore.ParameterSet.Config as cms

higgsToTauTauLeptonTauEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *')
)
higgsToTauTauLeptonTauEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('higgsToTauTauLeptonTauPath')
    )
)

