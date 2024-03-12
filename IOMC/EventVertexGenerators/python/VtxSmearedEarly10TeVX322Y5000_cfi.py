import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import Early10TeVX322Y5000VtxSmearingParameters,VtxSmearedCommon
VtxSmeared = cms.EDProducer("BetafuncEvtVtxGenerator",
    Early10TeVX322Y5000VtxSmearingParameters,
    VtxSmearedCommon
)



# foo bar baz
# 7CFenSpNClst6
# 7RQEDrDEtRW02
