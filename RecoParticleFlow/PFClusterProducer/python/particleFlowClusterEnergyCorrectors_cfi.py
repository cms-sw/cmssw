import FWCore.ParameterSet.Config as cms

emEnergyCorrector = cms.PSet(
    algoName = cms.string("PFClusterEMEnergyCorrector"),
    applyCrackCorrections = cms.bool(False)
    )
