import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import RealisticHIpPb2013CollisionVtxSmearingParameters,VtxSmearedCommon
VtxSmeared = cms.EDProducer("BetafuncEvtVtxGenerator",
    RealisticHIpPb2013CollisionVtxSmearingParameters,
    VtxSmearedCommon
)
