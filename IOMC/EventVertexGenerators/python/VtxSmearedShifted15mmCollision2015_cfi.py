import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *
VertexSmearingParameters = cms.PSet(
    Shifted15mmCollision2015VtxSmearingParameters,
    VtxSmearedCommon
)
