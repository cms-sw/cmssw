import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import Early10TeVX322Y10000VtxSmearingParameters,VtxSmearedCommon
VtxSmeared = cms.EDProducer("BetafuncEvtVtxGenerator",
    Early10TeVX322Y10000VtxSmearingParameters,
    VtxSmearedCommon
)



# foo bar baz
# 9c0r3EIRzVrQw
