import FWCore.ParameterSet.Config as cms

es_electronics_sim = cms.PSet(
    doESNoise = cms.bool(True),
    doFast = cms.bool(True)
)