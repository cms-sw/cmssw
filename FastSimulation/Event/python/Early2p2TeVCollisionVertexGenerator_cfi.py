import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *
myVertexGenerator = cms.PSet(
    Early2p2TeVCollisionVtxSmearingParameters,
    type = cms.string('BetaFunc')
    )
