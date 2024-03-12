import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *
VtxSmeared = cms.EDProducer("BetafuncEvtVtxGenerator",
    ShiftedCollision2015VtxSmearingParameters,
    VtxSmearedCommon
)
# foo bar baz
# wp2x8Gwa4Z4rn
