import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import Realistic25ns13p6TeVEarly2023CollisionVtxSmearingParameters,VtxSmearedCommon
VtxSmeared = cms.EDProducer("BetafuncEvtVtxGenerator",
    Realistic25ns13p6TeVEarly2023CollisionVtxSmearingParameters,
    VtxSmearedCommon
)
