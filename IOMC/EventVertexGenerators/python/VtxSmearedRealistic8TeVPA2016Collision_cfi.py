import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import Realistic8TeVPACollision2016VtxSmearingParameters,VtxSmearedCommon
VtxSmeared = cms.EDProducer("BetafuncEvtVtxGenerator",
    Realistic8TeVPACollision2016VtxSmearingParameters,
    VtxSmearedCommon
)
# foo bar baz
# U8MtfvBbF0m4L
# 0HwQO5WolEw7h
