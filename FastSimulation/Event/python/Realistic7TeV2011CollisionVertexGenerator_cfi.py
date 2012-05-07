import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *
myVertexGenerator = cms.PSet(
    Realistic7TeV2011CollisionVtxSmearingParameters,
    type = cms.string('BetaFunc')
)

