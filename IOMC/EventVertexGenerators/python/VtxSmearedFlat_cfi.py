import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import FlatVtxSmearingParameters,VtxSmearedCommon
VtxSmeared = cms.EDProducer("FlatEvtVtxGenerator",
    FlatVtxSmearingParameters,
    VtxSmearedCommon
)



# foo bar baz
# R3i3WypAi0bN0
