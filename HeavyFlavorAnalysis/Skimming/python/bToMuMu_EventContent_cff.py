import FWCore.ParameterSet.Config as cms

# keep essential info used to construct b to Jpsi
bToMuMuEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
bToMuMuEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('bToMuMuHLTPath')
    )
)

