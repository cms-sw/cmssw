import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import VtxSmearedCommon,NominalCollisionVtxSmearingParameters
VtxSmeared = cms.EDProducer("BetafuncEvtVtxGenerator",
    VtxSmearedCommon,
    NominalCollisionVtxSmearingParameters
)
