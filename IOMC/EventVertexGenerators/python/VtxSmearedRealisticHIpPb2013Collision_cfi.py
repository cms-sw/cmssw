import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import RealisticHIpPb2013CollisionVtxSmearingParameters,VtxSmearedCommon
VertexSmearingParameters = cms.PSet(
    RealisticHIpPb2013CollisionVtxSmearingParameters,
    VtxSmearedCommon
)
