import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *
VtxSmeared = cms.EDProducer("BetafuncEvtVtxGenerator",
    VtxSmearedCommon,
    Phi = cms.double(0.0),
    BetaStar = cms.double(150.0),
    Emittance = cms.double(0.90e-07),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(6.26),
    TimeOffset = cms.double(0.0),
    X0 = cms.double(0.244),
    Y0 = cms.double(0.393),
    Z0 = cms.double(0.41),
    Beta=cms.double(0.434)
)
