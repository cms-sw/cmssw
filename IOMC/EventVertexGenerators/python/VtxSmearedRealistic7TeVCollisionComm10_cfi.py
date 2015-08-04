import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import Realistic7TeVCollisionComm10VtxSmearingParameters,VtxSmearedCommon
VertexSmearingParameters = cms.PSet(
    Realistic7TeVCollisionComm10VtxSmearingParameters,
    VtxSmearedCommon
)



