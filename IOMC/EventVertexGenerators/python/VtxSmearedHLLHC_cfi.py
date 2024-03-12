import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *
VtxSmeared = cms.EDProducer("HLLHCEvtVtxGenerator",
    HLLHCVtxSmearingParameters,
    VtxSmearedCommon
)



# foo bar baz
# XP0TZf6UTUOq6
