import FWCore.ParameterSet.Config as cms

jpsiToMuMuEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)

jpsiToMuMuEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('jpsiToMuMuHLTPath')
    )
)

