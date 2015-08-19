import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import Realistic7TeV2011CollisionVtxSmearingParameters,VtxSmearedCommon
VertexSmearingParameters = cms.PSet(
    Realistic7TeV2011CollisionVtxSmearingParameters,
    VtxSmearedCommon
)



