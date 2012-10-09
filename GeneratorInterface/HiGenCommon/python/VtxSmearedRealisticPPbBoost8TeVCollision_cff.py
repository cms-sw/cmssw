import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *

Realistic5TeVCollisionPPbBoostVtxSmearingParameters= cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(1100.0),
    Emittance = cms.double(0.5e-07),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(6.889),
    TimeOffset = cms.double(0.0),
    X0 = cms.double(0.228489),
    Y0 = cms.double(0.4475616),
    Z0 = cms.double(0.224955),
    Beta=cms.double(-0.434)
)

VtxSmeared = cms.EDProducer("BetaBoostEvtVtxGenerator",
    VtxSmearedCommon,
    Realistic5TeVCollisionPPbBoostVtxSmearingParameters
)                           
