import FWCore.ParameterSet.Config as cms

jpsiToMuMuEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('jpsiToMuMuHLTPath')
    )
)

