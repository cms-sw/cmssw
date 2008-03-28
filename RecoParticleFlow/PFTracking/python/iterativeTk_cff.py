import FWCore.ParameterSet.Config as cms

from RecoTracker.Configuration.RecoTracker_cff import *
from RecoParticleFlow.PFTracking.first_cff import *
from RecoParticleFlow.PFTracking.second_cff import *
from RecoParticleFlow.PFTracking.third_cff import *
from RecoParticleFlow.PFTracking.fourth_cff import *
iterativeTk = cms.Sequence(first*second*third*fourth)

