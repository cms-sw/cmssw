import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFTracking.modules import PFElecTkProducer
pfTrackElec = PFElecTkProducer()

from Configuration.ProcessModifiers.egamma_lowPt_exclusive_cff import egamma_lowPt_exclusive
egamma_lowPt_exclusive.toModify(pfTrackElec,MinSCEnergy = 1.0)
