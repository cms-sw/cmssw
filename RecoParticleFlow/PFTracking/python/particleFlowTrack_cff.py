import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFTracking.pfNuclear_cfi import *
from RecoParticleFlow.PFTracking.pfV0_cfi import *


from RecoParticleFlow.PFTracking.pfTrackElec_cfi import *
particleFlowTrack = cms.Sequence(pfTrackElec)
particleFlowTrackWithNuclear = cms.Sequence(pfTrackElec*pfNuclear)
particleFlowTrackWithV0 = cms.Sequence(pfTrackElec*pfV0)

