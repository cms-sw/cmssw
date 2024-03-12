import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import Nominal2022PbPbCollisionVtxSmearingParameters,VtxSmearedCommon
VtxSmeared = cms.EDProducer("BetafuncEvtVtxGenerator",
    Nominal2022PbPbCollisionVtxSmearingParameters,
    VtxSmearedCommon
)
# foo bar baz
# E3XrYDnztXaCM
# QT5zOlDvIi8hS
