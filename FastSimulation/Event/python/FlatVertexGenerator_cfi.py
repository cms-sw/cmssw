import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *
myVertexGenerator = cms.PSet(
    FlatVtxSmearingParameters,
    type = cms.string('Flat')
)

