import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFTracking.pfTrack_cfi import *
from RecoParticleFlow.PFTracking.pfNuclear_cfi import *
from RecoParticleFlow.PFTracking.pfV0_cfi import *
from RecoParticleFlow.PFTracking.pfTrackElec_cfi import *
from RecoParticleFlow.PFTracking.particleFlowTrackWithDisplacedVertex_cff import *

particleFlowTrackTask = cms.Task(pfTrack, pfTrackElec)
particleFlowTrack = cms.Sequence(particleFlowTrackTask)

particleFlowTrackWithNuclearTask = cms.Task(pfTrack, pfTrackElec, pfNuclear)
particleFlowTrackWithNuclear = cms.Sequence(particleFlowTrackWithNuclearTask)

particleFlowTrackWithV0Task = cms.Task(pfTrack, pfTrackElec, pfV0)
particleFlowTrackWithV0 = cms.Sequence(particleFlowTrackWithV0Task)

pfTrackingGlobalRecoTask = cms.Task(particleFlowTrackWithDisplacedVertexTask)
pfTrackingGlobalReco = cms.Sequence(pfTrackingGlobalRecoTask)
