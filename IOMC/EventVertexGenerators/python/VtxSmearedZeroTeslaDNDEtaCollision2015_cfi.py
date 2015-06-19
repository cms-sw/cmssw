import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *
VtxSmeared = cms.EDProducer("BetafuncEvtVtxGenerator",
    ZeroTeslaDNDEtaCollision2015VtxSmearingParameters,
    VtxSmearedCommon
)
