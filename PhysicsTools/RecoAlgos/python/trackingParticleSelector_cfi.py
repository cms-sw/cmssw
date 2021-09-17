import FWCore.ParameterSet.Config as cms

import SimTracker.Common.trackingParticleSelector_cfi
trackingParticleSelector = SimTracker.Common.trackingParticleSelector_cfi.trackingParticleSelector.clone()

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(trackingParticleSelector, src = "mixData:MergedTrackTruth")
