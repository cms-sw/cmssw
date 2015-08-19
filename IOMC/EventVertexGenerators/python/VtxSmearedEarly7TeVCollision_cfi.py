import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import Early7TeVCollisionVtxSmearingParameters,VtxSmearedCommon
VertexSmearingParameters = cms.PSet(
    Early7TeVCollisionVtxSmearingParameters,
    VtxSmearedCommon
)



