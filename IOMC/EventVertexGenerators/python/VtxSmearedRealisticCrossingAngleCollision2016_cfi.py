import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import VtxSmearedCommon

Realistic25ns13TeV2016PreTS2CollisionVtxSmearingParameters = cms.PSet(
    vertexSize = cms.double(10.0e-6), # in m
    beamDivergence = cms.double(20.0e-6), # in rad
    scatteringAngle = cms.double(25.0e-6), # in rad

    # switches
    simulateVertexX = cms.bool(True),
    simulateVertexY = cms.bool(True),
    simulateScatteringAngleX = cms.bool(True),
    simulateScatteringAngleY = cms.bool(True),
    simulateBeamDivergence = cms.bool(True),

    # crossing angle
    halfCrossingAngleSector45 = cms.double(179.394e-6), # in rad
    halfCrossingAngleSector56 = cms.double(191.541e-6), # in rad
)

VtxSmeared = cms.EDProducer('CrossingAngleVtxGenerator',
    Realistic25ns13TeV2016PreTS2CollisionVtxSmearingParameters,
    VtxSmearedCommon
)
