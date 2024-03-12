import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import Fixed_EmitRealistic5TeVppCollision2017VtxSmearingParameters,VtxSmearedCommon
VtxSmeared = cms.EDProducer("BetafuncEvtVtxGenerator",
    Fixed_EmitRealistic5TeVppCollision2017VtxSmearingParameters,
    VtxSmearedCommon
)
# foo bar baz
# 8ap3VxqLKUlN3
# NzW5tl5BJ59BX
