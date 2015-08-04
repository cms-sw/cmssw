import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import Realistic2p76TeV2011CollisionVtxSmearingParameters,VtxSmearedCommon
VertexSmearingParameters = cms.PSet(
    Realistic2p76TeV2011CollisionVtxSmearingParameters,
    VtxSmearedCommon
)



