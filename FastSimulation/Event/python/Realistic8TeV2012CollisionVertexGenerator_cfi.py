import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *
myVertexGenerator = cms.PSet(
    Realistic8TeV2012CollisionVtxSmearingParameters,
    type = cms.string('BetaFunc')
)

