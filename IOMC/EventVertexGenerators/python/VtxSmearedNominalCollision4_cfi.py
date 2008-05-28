import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *
VtxSmeared = cms.EDFilter("BetafuncEvtVtxGenerator",
    NominalCollision4VtxSmearingParameters,
    VtxSmearedCommon
)



