import FWCore.ParameterSet.Config as cms

upsilonToMuMuEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('upsilonToMuMuHLTPath')
    )
)

