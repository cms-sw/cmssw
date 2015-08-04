import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import FlatVtxSmearingParameters,VtxSmearedCommon
VertexSmearingParameters = cms.PSet(
    FlatVtxSmearingParameters,
    VtxSmearedCommon
)



