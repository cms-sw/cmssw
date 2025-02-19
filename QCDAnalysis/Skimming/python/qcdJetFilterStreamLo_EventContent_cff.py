import FWCore.ParameterSet.Config as cms

qcdJetFilterStreamLoEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('qcdJetFilterStreamLoPath', 
            'HLT1jet:HLT')
    )
)

