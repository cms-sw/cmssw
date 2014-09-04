import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import Early10TeVX322Y5000VtxSmearingParameters,VtxSmearedCommon
VtxSmeared = cms.EDProducer("BetafuncEvtVtxGenerator",
    Early10TeVX322Y5000VtxSmearingParameters,
    VtxSmearedCommon
)



