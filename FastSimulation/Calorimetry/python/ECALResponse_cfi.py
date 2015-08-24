import FWCore.ParameterSet.Config as cms

ECALScaleBlock = cms.PSet(
    ECALResponseScaling = cms.PSet(
        fileName = cms.untracked.string("FastSimulation/Calorimetry/data/scaleECALFastsim.root"),
        histogramName = cms.untracked.string("scaleVsEVsEta"),
        interpolate3D = cms.untracked.bool(False)
    )
)



