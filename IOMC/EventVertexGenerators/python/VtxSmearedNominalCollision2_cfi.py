import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import VtxSmearedCommon,NominalCollision2VtxSmearingParameters
VertexSmearingParameters = cms.PSet(
    NominalCollision2VtxSmearingParameters,
    VtxSmearedCommon
)



