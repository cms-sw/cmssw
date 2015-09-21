import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import UpdatedHICollision2015VtxSmearingParameters,VtxSmearedCommon
VtxSmeared = cms.EDProducer("BetafuncEvtVtxGenerator",
    UpdatedHICollision2015VtxSmearingParameters,
    VtxSmearedCommon
)



