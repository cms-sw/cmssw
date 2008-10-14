import FWCore.ParameterSet.Config as cms
import copy

#
from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *
from RecoJets.JetAssociationProducers.ic5PFJetTracksAssociatorAtVertex_cfi import *
#PFTauTagInfo Producer
from RecoTauTag.RecoTau.PFRecoTauTagInfoProducer_cfi import *
from RecoTauTag.RecoTau.PFRecoTauProducer_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByIsolation_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByLeadingTrackFinding_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByLeadingTrackPtCut_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstElectron_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstMuon_cfi import *

from RecoTauTag.RecoTau.PFRecoTauHighEfficiency_cff import *


PFTau = cms.Sequence(
    ic5PFJetTracksAssociatorAtVertex*
    pfRecoTauTagInfoProducer*
    pfRecoTauProducer*
    pfRecoTauDiscriminationByIsolation*
    pfRecoTauDiscriminationHighEfficiency*
    pfRecoTauDiscriminationByLeadingTrackFinding*
    pfRecoTauDiscriminationByLeadingTrackPtCut*
    pfRecoTauDiscriminationAgainstElectron*
    pfRecoTauDiscriminationAgainstMuon
    )


PFTauHighEfficiency = cms.Sequence(
    ic5PFJetTracksAssociatorAtVertex*
    pfRecoTauTagInfoProducer*
    pfRecoTauProducerHighEfficiency*
    pfRecoTauDiscriminationByIsolationHighEfficiency*
    pfRecoTauDiscriminationByLeadingTrackFindingHighEfficiency*
    pfRecoTauDiscriminationByLeadingTrackPtCutHighEfficiency*
    pfRecoTauDiscriminationAgainstElectronHighEfficiency*
    pfRecoTauDiscriminationAgainstMuonHighEfficiency
    )
