import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *

VtxSmeared = cms.EDProducer("BetaBoostEvtVtxGenerator",
    VtxSmearedCommon,
    Realistic5TeVPACollision2016VtxSmearingParameters,
    Beta=cms.double(0.434)
)
