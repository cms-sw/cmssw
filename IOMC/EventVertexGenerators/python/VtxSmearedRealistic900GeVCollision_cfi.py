import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import Realistic900GeVCollisionVtxSmearingParameters,VtxSmearedCommon
VtxSmeared = cms.EDProducer("BetafuncEvtVtxGenerator",
    Realistic900GeVCollisionVtxSmearingParameters,
    VtxSmearedCommon
)



# foo bar baz
# FkZanZnER1R0V
