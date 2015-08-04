import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *
VertexSmearingParameters = cms.PSet(
    Shifted5mmCollision2015VtxSmearingParameters,
    VtxSmearedCommon
)
