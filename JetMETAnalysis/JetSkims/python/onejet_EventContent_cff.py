import FWCore.ParameterSet.Config as cms

onejetEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
onejetEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('onejetHLTPath', 
            'onejetpe1HLTPath', 
            'onejetpe3HLTPath', 
            'onejetpe5HLTPath', 
            'onejetpe7HLTPath')
    )
)

