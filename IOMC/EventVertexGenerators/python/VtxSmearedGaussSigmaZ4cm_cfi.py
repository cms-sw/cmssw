import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import GaussVtxSigmaZ4cmSmearingParameters,VtxSmearedCommon
VtxSmeared = cms.EDProducer("GaussEvtVtxGenerator",
    GaussVtxSigmaZ4cmSmearingParameters,
    VtxSmearedCommon
)



