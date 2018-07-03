import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import Realistic5TeVPACollision2016VtxSmearingParameters,VtxSmearedCommon
VtxSmeared = cms.EDProducer("BetafuncEvtVtxGenerator",
    Realistic5TeVPACollision2016VtxSmearingParameters,
    VtxSmearedCommon
)
