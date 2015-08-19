import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import Early10TeVX322Y250VtxSmearingParameters,VtxSmearedCommon
VertexSmearingParameters = cms.PSet(
    Early10TeVX322Y250VtxSmearingParameters,
    VtxSmearedCommon
)



