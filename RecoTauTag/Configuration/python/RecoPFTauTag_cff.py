import FWCore.ParameterSet.Config as cms
import copy

from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *
from RecoJets.JetAssociationProducers.ic5PFJetTracksAssociatorAtVertex_cfi import *
#PFTauTagInfo Producer
from RecoTauTag.RecoTau.PFRecoTauTagInfoProducer_cfi import *
from RecoTauTag.RecoTau.PFRecoTauProducer_cfi import *

from RecoTauTag.RecoTau.PFRecoTauHighEfficiency_cff import *

from RecoTauTag.Configuration.RecoPFTauTag_cfi import *


PFTau = cms.Sequence(
    ic5PFJetTracksAssociatorAtVertex*
    pfRecoTauTagInfoProducer*
    pfRecoTauProducer*
    PFTauDiscrimination
)


PFTauHighEfficiency = cms.Sequence(
    ic5PFJetTracksAssociatorAtVertex*
    pfRecoTauTagInfoProducer*
    pfRecoTauProducerHighEfficiency*
    PFTauDiscriminationHighEfficiency
)
