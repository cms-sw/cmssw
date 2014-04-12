import FWCore.ParameterSet.Config as cms

# Tau tagging

from RecoTauTag.Configuration.RecoPFTauTag_cff import *
#recoTauAK5PFJets08Region.pfSrc = cms.InputTag("FSparticleFlow") # AG
from RecoTauTag.RecoTau.RecoTauShrinkingConeProducer_cfi import _shrinkingConeRecoTausConfig
#_shrinkingConeRecoTausConfig.pfCandSrc = cms.InputTag("FSparticleFlow") # AG
from RecoTauTag.RecoTau.PFRecoTauTagInfoProducer_cfi import pfRecoTauTagInfoProducer
#pfRecoTauTagInfoProducer.PFCandidateProducer = cms.InputTag("FSparticleFlow") # AG
from RecoTauTag.RecoTau.RecoTauCombinatoricProducer_cfi import _combinatoricTauConfig
#_combinatoricTauConfig.pfCandSrc = cms.InputTag("FSparticleFlow") # AG

famosPFTauTaggingSequence = cms.Sequence(PFTau)
    









