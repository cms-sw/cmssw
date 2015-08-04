import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import Realistic7TeVCollision2010BVtxSmearingParameters,VtxSmearedCommon
VertexSmearingParameters = cms.PSet(
    Realistic7TeVCollision2010BVtxSmearingParameters,
    VtxSmearedCommon
)



