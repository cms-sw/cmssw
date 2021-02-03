import FWCore.ParameterSet.Config as cms

egammaCorrections = cms.PSet(
    value = cms.vdouble(
        1.0, 1.0, 1.01, 1.01, 1.02,
        1.01, 1.01, 1.01
    )
)