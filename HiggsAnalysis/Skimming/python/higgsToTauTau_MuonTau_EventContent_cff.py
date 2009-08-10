import FWCore.ParameterSet.Config as cms

higgsToTauTauMuonTauEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *')
)
higgsToTauTauMuonTauEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('higgsToTauTauMuonTauPath')
    )
)

