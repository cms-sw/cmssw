import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import NominalCollision3VtxSmearingParameters,VtxSmearedCommon
VertexSmearingParameters = cms.PSet(
    NominalCollision3VtxSmearingParameters,
    VtxSmearedCommon
)



