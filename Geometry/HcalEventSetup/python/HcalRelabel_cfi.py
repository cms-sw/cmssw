import FWCore.ParameterSet.Config as cms

HcalReLabel = cms.PSet( 
    RelabelHits  = cms.untracked.bool(False),
    RelabelRules = cms.untracked.PSet(
        CorrectPhi = cms.untracked.bool(False),
    )
)
