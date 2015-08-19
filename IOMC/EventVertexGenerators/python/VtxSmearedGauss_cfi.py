import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import GaussVtxSmearingParameters,VtxSmearedCommon
VertexSmearingParameters = cms.PSet(
    GaussVtxSmearingParameters,
    VtxSmearedCommon
)



