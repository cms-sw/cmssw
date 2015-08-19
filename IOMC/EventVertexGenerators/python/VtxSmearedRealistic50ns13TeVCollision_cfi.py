import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import Realistic50ns13TeVCollisionVtxSmearingParameters,VtxSmearedCommon
VertexSmearingParameters = cms.PSet(
    Realistic50ns13TeVCollisionVtxSmearingParameters,
    VtxSmearedCommon
)
