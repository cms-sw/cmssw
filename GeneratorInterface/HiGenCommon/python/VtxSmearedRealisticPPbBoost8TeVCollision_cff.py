import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *

Realistic5TeVCollisionPPbBoostVtxSmearingParameters= cms.PSet(

    Phi = cms.double(0.0),
    BetaStar = cms.double(80.0),
    Emittance = cms.double(6.25e-07),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(8.0),
    TimeOffset = cms.double(0.0),
    X0 = cms.double(0.2440),
    Y0 = cms.double(0.3929),
    Z0 = cms.double(0.4145)
    Beta=cms.double(-0.434)
)

VtxSmeared = cms.EDProducer("BetaBoostEvtVtxGenerator",
    VtxSmearedCommon,
    Realistic5TeVCollisionPPbBoostVtxSmearingParameters
)                           
