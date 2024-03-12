import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import *
VtxSmeared = cms.EDProducer("BetafuncEvtVtxGenerator",
    Realistic50ns13TeVCollisionZeroTeslaVtxSmearingParameters,
    VtxSmearedCommon
)
# foo bar baz
# CfOdCi1KPU3ua
# 2HOsFK9AKLf87
