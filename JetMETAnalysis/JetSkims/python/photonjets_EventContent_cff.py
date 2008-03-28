import FWCore.ParameterSet.Config as cms

photonjetsEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
photonjetsEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('singlePhotonHLTPath', 'singleRelaxedPhotonHLTPath', 'singlePhotonHLTPath12')
    )
)

