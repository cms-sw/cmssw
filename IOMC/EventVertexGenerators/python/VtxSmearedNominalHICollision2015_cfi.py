import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import NominalHICollision2015VtxSmearingParameters,VtxSmearedCommon
VertexSmearingParameters = cms.PSet(
    NominalHICollision2015VtxSmearingParameters,
    VtxSmearedCommon
)



