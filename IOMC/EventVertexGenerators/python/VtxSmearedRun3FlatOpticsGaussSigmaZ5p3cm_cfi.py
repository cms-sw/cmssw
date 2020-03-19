import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import Run3FlatOpticsGaussVtxSigmaZ5p3cmSmearingParameters,VtxSmearedCommon
VtxSmeared = cms.EDProducer("GaussEvtVtxGenerator",
    Run3FlatOpticsGaussVtxSigmaZ5p3cmSmearingParameters,
    VtxSmearedCommon
)
