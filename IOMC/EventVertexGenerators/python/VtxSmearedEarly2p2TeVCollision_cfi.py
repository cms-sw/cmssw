import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import Early2p2TeVCollisionVtxSmearingParameters,VtxSmearedCommon
VtxSmeared = cms.EDProducer("BetafuncEvtVtxGenerator",
    Early2p2TeVCollisionVtxSmearingParameters,
    VtxSmearedCommon
)



# foo bar baz
# 17KatQpsCQl47
