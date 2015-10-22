import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import Nominal5TeVpp2015VtxSmearingParameters,VtxSmearedCommon
VtxSmeared = cms.EDProducer("BetafuncEvtVtxGenerator",
    Nominal5TeVpp2015VtxSmearingParameters,
    VtxSmearedCommon
)
