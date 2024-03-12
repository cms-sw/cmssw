import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import GaussVtxSmearingParameters,VtxSmearedCommon
VtxSmeared = cms.EDProducer("GaussEvtVtxGenerator",
    GaussVtxSmearingParameters,
    VtxSmearedCommon
)



# foo bar baz
# mf6lfyKRTV8OO
# SRdPxvJgOogzN
