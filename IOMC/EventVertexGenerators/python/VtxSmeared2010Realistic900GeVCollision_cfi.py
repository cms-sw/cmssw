import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *
VtxSmeared = cms.EDProducer("BetafuncEvtVtxGenerator",
    2010Realistic900GeVCollisionVtxSmearingParameters,
    VtxSmearedCommon
)



