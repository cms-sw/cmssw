import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *
myVertexGenerator = cms.PSet(
    Realistic8TeVCollisionVtxSmearingParameters,
    type = cms.string('BetaFunc')
)

