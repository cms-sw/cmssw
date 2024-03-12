import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import NominalCollision4VtxSmearingParameters,VtxSmearedCommon
VtxSmeared = cms.EDProducer("BetafuncEvtVtxGenerator",
    NominalCollision4VtxSmearingParameters,
    VtxSmearedCommon
)



# foo bar baz
# g4PxWAbdsyANf
# rtFK1rfY0sYIN
