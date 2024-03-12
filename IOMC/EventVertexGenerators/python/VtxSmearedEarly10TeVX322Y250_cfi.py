import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import Early10TeVX322Y250VtxSmearingParameters,VtxSmearedCommon
VtxSmeared = cms.EDProducer("BetafuncEvtVtxGenerator",
    Early10TeVX322Y250VtxSmearingParameters,
    VtxSmearedCommon
)



# foo bar baz
# B3AkER3ls2gaf
# upwpKL6f1zRhB
