import FWCore.ParameterSet.Config as cms

sumETEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
sumETEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('sumETHLTPath')
    )
)

