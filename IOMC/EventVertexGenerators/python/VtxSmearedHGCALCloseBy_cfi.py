import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *
VtxSmeared = cms.EDProducer("GaussEvtVtxGenerator",
    GaussVtxSmearingParameters,
    VtxSmearedCommon
)
VtxSmeared.SigmaX = cms.double(0)
VtxSmeared.SigmaY = cms.double(0)
VtxSmeared.SigmaZ = cms.double(0)
