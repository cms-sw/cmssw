import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import VtxSmearedCommon,EarlyCollisionVtxSmearingParameters
VtxSmeared = cms.EDProducer("BetafuncEvtVtxGenerator",
    VtxSmearedCommon,
    EarlyCollisionVtxSmearingParameters
)



