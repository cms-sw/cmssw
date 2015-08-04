import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import Early2p2TeVCollisionVtxSmearingParameters,VtxSmearedCommon
VertexSmearingParameters = cms.PSet(
    Early2p2TeVCollisionVtxSmearingParameters,
    VtxSmearedCommon
)



