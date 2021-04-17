import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFClusterProducer.particleFlowClusterTimeAssignerDefault_cfi import *

particleFlowTimeAssignerECAL = particleFlowClusterTimeAssignerDefault.clone(
    timeSrc     = 'ecalBarrelClusterFastTimer:PerfectResolutionModel',
    timeResoSrc = 'ecalBarrelClusterFastTimer:PerfectResolutionModelResolution'
)

