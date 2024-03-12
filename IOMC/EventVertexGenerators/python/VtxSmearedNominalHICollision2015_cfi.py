import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import NominalHICollision2015VtxSmearingParameters,VtxSmearedCommon
VtxSmeared = cms.EDProducer("BetafuncEvtVtxGenerator",
    NominalHICollision2015VtxSmearingParameters,
    VtxSmearedCommon
)



# foo bar baz
# lEoMni1uTd16W
# Jfgw2nkQ8QjSg
