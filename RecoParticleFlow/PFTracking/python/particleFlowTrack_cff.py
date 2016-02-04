import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFTracking.pfTrack_cfi import *
from RecoParticleFlow.PFTracking.pfNuclear_cfi import *
from RecoParticleFlow.PFTracking.pfV0_cfi import *
from RecoParticleFlow.PFTracking.pfTrackElec_cfi import *


particleFlowTrack = cms.Sequence(pfTrack*pfTrackElec)
particleFlowTrackWithNuclear = cms.Sequence(pfTrack*pfTrackElec*pfNuclear)
particleFlowTrackWithV0 = cms.Sequence(pfTrack*pfTrackElec*pfV0)

