import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import NominalCollision4VtxSmearingParameters,VtxSmearedCommon
VertexSmearingParameters = cms.PSet(
    NominalCollision4VtxSmearingParameters,
    VtxSmearedCommon
)



