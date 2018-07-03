import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFClusterProducer.particleFlowClusterTimeAssignerDefault_cfi import *

particleFlowTimeAssignerECAL = particleFlowClusterTimeAssignerDefault.clone()
particleFlowTimeAssignerECAL.timeSrc = cms.InputTag('ecalBarrelClusterFastTimer:PerfectResolutionModel')
particleFlowTimeAssignerECAL.timeResoSrc = cms.InputTag('ecalBarrelClusterFastTimer:PerfectResolutionModelResolution')


