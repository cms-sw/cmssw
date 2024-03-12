import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import NominalCollision1VtxSmearingParameters,VtxSmearedCommon
VtxSmeared = cms.EDProducer("BetafuncEvtVtxGenerator",
    NominalCollision1VtxSmearingParameters,
    VtxSmearedCommon
)



# foo bar baz
# DGeqbAGkvdejR
# 7Wxd2KtWDlC9l
