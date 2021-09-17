import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFTracking.pfTrackElec_cfi import *

from RecoParticleFlow.PFTracking.pfConversions_cfi import *

particleFlowTrackWithConversionTask =cms.Task(
    pfTrackElec,
    pfConversions
    )
particleFlowTrackWithConversion =cms.Sequence(particleFlowTrackWithConversionTask)
