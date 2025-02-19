import FWCore.ParameterSet.Config as cms

qcdJetFilterStreamHiEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('qcdJetFilterStreamHiPath')
    )
)

