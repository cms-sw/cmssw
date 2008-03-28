import FWCore.ParameterSet.Config as cms

metHighEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
metHighEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('metHighHLTPath')
    )
)

