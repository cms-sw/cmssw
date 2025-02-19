import FWCore.ParameterSet.Config as cms

metLowEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
metLowEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('metLowPre1HLTPath', 
            'metLowPre2HLTPath', 
            'metLowPre3HLTPath')
    )
)

