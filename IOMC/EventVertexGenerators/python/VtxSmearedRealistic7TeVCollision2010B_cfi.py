import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import Realistic7TeVCollision2010BVtxSmearingParameters,VtxSmearedCommon
VtxSmeared = cms.EDProducer("BetafuncEvtVtxGenerator",
    Realistic7TeVCollision2010BVtxSmearingParameters,
    VtxSmearedCommon
)



# foo bar baz
# 7hEHnCVHE9Suu
# a8XjPeYiN8j9f
