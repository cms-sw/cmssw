import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import Realistic7TeVCollisionVtxSmearingParameters,VtxSmearedCommon
VtxSmeared = cms.EDProducer("BetafuncEvtVtxGenerator",
    Realistic7TeVCollisionVtxSmearingParameters,
    VtxSmearedCommon
)



# foo bar baz
# rOBw91yaQX0MD
# qIXEpamp4l1QL
