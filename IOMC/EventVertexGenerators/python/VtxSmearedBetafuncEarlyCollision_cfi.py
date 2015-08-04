import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import VtxSmearedCommon,EarlyCollisionVtxSmearingParameters
VertexSmearingParameters = cms.PSet(
    VtxSmearedCommon,
    EarlyCollisionVtxSmearingParameters
)



