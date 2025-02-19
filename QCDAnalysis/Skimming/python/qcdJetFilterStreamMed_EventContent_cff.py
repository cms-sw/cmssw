import FWCore.ParameterSet.Config as cms

qcdJetFilterStreamMedEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('qcdJetFilterStreamMedPath')
    )
)

