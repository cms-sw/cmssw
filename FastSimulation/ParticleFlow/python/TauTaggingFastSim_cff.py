import FWCore.ParameterSet.Config as cms

# Tau tagging

from RecoJets.JetAssociationProducers.ic5JetTracksAssociatorAtVertex_cfi import *
from RecoJets.JetAssociationProducers.ic5PFJetTracksAssociatorAtVertex_cfi import *
ic5JetTracksAssociatorAtVertex.tracks = 'generalTracks'
ic5PFJetTracksAssociatorAtVertex.tracks = 'generalTracks'
from RecoTauTag.Configuration.RecoTauTag_cff import *

from RecoTauTag.Configuration.RecoPFTauTag_cff import *

famosTauTaggingSequence = cms.Sequence(tautagging)
famosPFTauTaggingSequence = cms.Sequence(PFTau)
    









