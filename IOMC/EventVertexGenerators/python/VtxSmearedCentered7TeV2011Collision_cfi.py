import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import Centered7TeV2011CollisionVtxSmearingParameters,VtxSmearedCommon
VertexSmearingParameters = cms.PSet(
    Centered7TeV2011CollisionVtxSmearingParameters,
    VtxSmearedCommon
)



