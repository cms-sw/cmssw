import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import RealisticHI2011CollisionVtxSmearingParameters,VtxSmearedCommon
VtxSmeared = cms.EDProducer("BetafuncEvtVtxGenerator",
    RealisticHI2011CollisionVtxSmearingParameters,
    VtxSmearedCommon
)



# foo bar baz
# 2IGSMGEbR9IDn
# ri2rDvtZCYveA
