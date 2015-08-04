import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import NominalCollision1VtxSmearingParameters,VtxSmearedCommon
VertexSmearingParameters = cms.PSet(
    NominalCollision1VtxSmearingParameters,
    VtxSmearedCommon
)



