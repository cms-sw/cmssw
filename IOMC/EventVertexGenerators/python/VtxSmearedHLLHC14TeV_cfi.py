import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *
HLLHCVtxSmearingParameters14TeV = HLLHCVtxSmearingParameters.clone( EprotonInGeV = cms.double(7000) )
VtxSmeared = cms.EDProducer("HLLHCEvtVtxGenerator",
    HLLHCVtxSmearingParameters14TeV,
    VtxSmearedCommon
)



