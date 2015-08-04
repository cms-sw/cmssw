import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *
VertexSmearingParameters = cms.PSet(
    NominalCollision2015VtxSmearingParameters,
    VtxSmearedCommon
)
