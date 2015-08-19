import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import Early10TeVX322Y1000VtxSmearingParameters,VtxSmearedCommon
VertexSmearingParameters = cms.PSet(
    Early10TeVX322Y1000VtxSmearingParameters,
    VtxSmearedCommon
)



