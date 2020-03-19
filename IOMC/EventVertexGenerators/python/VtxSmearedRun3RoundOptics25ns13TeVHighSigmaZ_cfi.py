import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *
VtxSmeared = cms.EDProducer("BetafuncEvtVtxGenerator",
    Run3RoundOptics25ns13TeVHighSigmaZVtxSmearingParameters,
    VtxSmearedCommon
)