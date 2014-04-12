import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFTracking.pfTrack_cfi import *
from RecoParticleFlow.PFTracking.pfTrackElec_cfi import *
from RecoParticleFlow.PFTracking.pfDisplacedTrackerVertex_cfi import *
from RecoParticleFlow.PFTracking.pfConversions_cfi import *
from RecoParticleFlow.PFTracking.pfV0_cfi import *
from RecoParticleFlow.PFTracking.particleFlowDisplacedVertexCandidate_cff import *
from RecoParticleFlow.PFTracking.particleFlowDisplacedVertex_cff import *



particleFlowTrackWithDisplacedVertex =cms.Sequence(
    pfTrack*
    pfConversions*
    pfV0*
    particleFlowDisplacedVertexCandidate*
    particleFlowDisplacedVertex*
    pfDisplacedTrackerVertex*
    pfTrackElec
    )


