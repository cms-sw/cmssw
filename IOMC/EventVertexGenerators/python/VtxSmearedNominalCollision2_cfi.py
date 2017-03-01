import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import VtxSmearedCommon,NominalCollision2VtxSmearingParameters
VtxSmeared = cms.EDProducer("BetafuncEvtVtxGenerator",
    VtxSmearedCommon,
    NominalCollision2VtxSmearingParameters
)



