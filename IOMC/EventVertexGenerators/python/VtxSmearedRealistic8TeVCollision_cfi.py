import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import Realistic8TeVCollisionVtxSmearingParameters,VtxSmearedCommon
VtxSmeared = cms.EDProducer("BetafuncEvtVtxGenerator",
    Realistic8TeVCollisionVtxSmearingParameters,
    VtxSmearedCommon
)



# foo bar baz
# mw8adBc1zduNo
