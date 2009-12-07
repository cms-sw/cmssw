import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *
myVertexGenerator = cms.PSet(
        Early900GeVCollisionVtxSmearingParameters,
        type = cms.string('BetaFunc')
        )

