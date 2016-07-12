import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *

VtxSmeared = cms.EDProducer("BetaBoostEvtVtxGenerator",
    VtxSmearedCommon,
    Realistic50ns13TeVCollisionVtxSmearingParameters,
    Beta=cms.double(-0.434)
)
