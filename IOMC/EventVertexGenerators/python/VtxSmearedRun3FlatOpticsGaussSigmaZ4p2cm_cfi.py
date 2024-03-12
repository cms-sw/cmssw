import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import Run3FlatOpticsGaussVtxSigmaZ4p2cmSmearingParameters,VtxSmearedCommon
VtxSmeared = cms.EDProducer("GaussEvtVtxGenerator",
    Run3FlatOpticsGaussVtxSigmaZ4p2cmSmearingParameters,
    VtxSmearedCommon
)
# foo bar baz
# 8MSXoblsYwi9g
# EnqI0SeaI6ROU
