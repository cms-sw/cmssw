import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import Nominal2025OOCollisionVtxSmearingParameters,VtxSmearedCommon
VtxSmeared = cms.EDProducer("BetafuncEvtVtxGenerator",
    Nominal2025OOCollisionVtxSmearingParameters,
    VtxSmearedCommon
)
