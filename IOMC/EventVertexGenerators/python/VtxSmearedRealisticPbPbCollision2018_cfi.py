import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import RealisticPbPbCollision2018VtxSmearingParameters,VtxSmearedCommon
VtxSmeared = cms.EDProducer("BetafuncEvtVtxGenerator",
    RealisticPbPbCollision2018VtxSmearingParameters,
    VtxSmearedCommon
)
