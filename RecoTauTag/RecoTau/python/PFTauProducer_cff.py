import FWCore.ParameterSet.Config as cms

from RecoJets.JetAssociationProducers.ic5PFJetTracksAssociatorAtVertex_cfi import *
from RecoTauTag.RecoTau.PFRecoTauTagInfoProducer_cfi import *
from RecoTauTag.RecoTau.PFRecoTauProducer_cfi import *
PFTau = cms.Sequence(cms.SequencePlaceholder("particleFlowJetCandidates")*cms.SequencePlaceholder("iterativeCone5PFJets")*ic5PFJetTracksAssociatorAtVertex*pfRecoTauTagInfoProducer*pfRecoTauProducer)

