import FWCore.ParameterSet.Config as cms

upsilonToMuMuEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)

upsilonToMuMuEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('upsilonToMuMuHLTPath')
    )
)

