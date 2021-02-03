import FWCore.ParameterSet.Config as cms

ecal_electronics_sim = cms.PSet(
    ConstantTerm = cms.double(0.003),
    applyConstantTerm = cms.bool(True),
    doENoise = cms.bool(True)
)