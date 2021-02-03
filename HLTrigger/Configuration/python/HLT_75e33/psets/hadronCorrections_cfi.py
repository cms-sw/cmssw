import FWCore.ParameterSet.Config as cms

hadronCorrections = cms.PSet(
    value = cms.vdouble(
        1.28, 1.28, 1.24, 1.19, 1.17,
        1.17, 1.17, 1.17
    )
)