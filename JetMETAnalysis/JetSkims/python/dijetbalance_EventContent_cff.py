import FWCore.ParameterSet.Config as cms

dijetbalanceEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
dijetbalanceEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('dijetbalance30HLTPath', 
            'dijetbalance60HLTPath', 
            'dijetbalance110HLTPath', 
            'dijetbalance150HLTPath', 
            'dijetbalance200HLTPath')
    )
)

