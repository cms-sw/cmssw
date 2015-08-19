import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import Realistic8TeV2012CollisionVtxSmearingParameters,VtxSmearedCommon
VertexSmearingParameters = cms.PSet(
    Realistic8TeV2012CollisionVtxSmearingParameters,
    VtxSmearedCommon
)



