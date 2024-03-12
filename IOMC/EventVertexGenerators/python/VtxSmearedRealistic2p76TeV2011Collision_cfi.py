import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import Realistic2p76TeV2011CollisionVtxSmearingParameters,VtxSmearedCommon
VtxSmeared = cms.EDProducer("BetafuncEvtVtxGenerator",
    Realistic2p76TeV2011CollisionVtxSmearingParameters,
    VtxSmearedCommon
)



# foo bar baz
# YYzUSno2v7kEX
