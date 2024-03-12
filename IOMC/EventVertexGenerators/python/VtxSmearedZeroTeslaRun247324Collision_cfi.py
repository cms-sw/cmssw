import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *
VtxSmeared = cms.EDProducer("BetafuncEvtVtxGenerator",
    ZeroTeslaRun247324CollisionVtxSmearingParameters,
    VtxSmearedCommon
)
# foo bar baz
# d0dJit540xHET
