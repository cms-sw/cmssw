import FWCore.ParameterSet.Config as cms
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByCharge_cfi          import *



import RecoTauTag.RecoTau.PFRecoTauDiscriminationByTrackIsolation_cfi 
lowptpfTauDiscrByTrackIsolation = RecoTauTag.RecoTau.PFRecoTauDiscriminationByTrackIsolation_cfi.pfRecoTauDiscriminationByTrackIsolation.clone()
lowptpfTauDiscrByTrackIsolation.PFTauProducer  = cms.InputTag('pfRecoTauProducerHighEfficiency')
lowptpfTauDiscrByTrackIsolation.ManipulateTracks_insteadofChargedHadrCands = cms.bool(True)
lowptpfTauDiscrByTrackIsolation.maxChargedPt                               = cms.double(1.)
lowptpfTauDiscrByTrackIsolation.SumOverCandidates                          = cms.bool(True)
lowptpfTauDiscrByTrackIsolation.ApplyDiscriminationByTrackerIsolation      = cms.bool(True)


lowptpfTauDiscrByRelTrackIsolation = RecoTauTag.RecoTau.PFRecoTauDiscriminationByTrackIsolation_cfi.pfRecoTauDiscriminationByTrackIsolation.clone()
lowptpfTauDiscrByRelTrackIsolation.PFTauProducer  = cms.InputTag('pfRecoTauProducerHighEfficiency')
lowptpfTauDiscrByRelTrackIsolation.ManipulateTracks_insteadofChargedHadrCands = cms.bool(True)
lowptpfTauDiscrByRelTrackIsolation.maxChargedPt                               = cms.double(0.05)
lowptpfTauDiscrByRelTrackIsolation.SumOverCandidates                          = cms.bool(True)
lowptpfTauDiscrByRelTrackIsolation.ApplyDiscriminationByTrackerIsolation      = cms.bool(True)
lowptpfTauDiscrByRelTrackIsolation.TrackIsolationOverTauPt                    = cms.bool(False)

import  RecoTauTag.TauTagTools.PFRecoTauLogicalDiscriminator_cfi  
lowptcombDiscr = RecoTauTag.TauTagTools.PFRecoTauLogicalDiscriminator_cfi.pfRecoTauLogicalDiscriminator.clone()
lowptcombDiscr.TauDiscriminators=(
    'pfRecoTauDiscriminationByCharge',
    'lowptpfTauDiscrByTrackIsolation',
    'lowptpfTauDiscrByRelTrackIsolation'
    )
PFRecoTauLowPt = cms.Sequence(pfRecoTauDiscriminationByCharge*
                              lowptpfTauDiscrByTrackIsolation*
                              lowptpfTauDiscrByRelTrackIsolation*
                              lowptcombDiscr
                              )
