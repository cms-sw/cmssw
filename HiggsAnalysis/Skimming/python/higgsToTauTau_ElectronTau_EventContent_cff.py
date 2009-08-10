import FWCore.ParameterSet.Config as cms

higgsToTauTauElectronTauEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *')
)
higgsToTauTauElectronTauEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('higgsToTauTauElectronTauPath')
    )
)

