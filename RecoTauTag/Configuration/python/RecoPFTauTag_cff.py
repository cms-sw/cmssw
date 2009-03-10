import FWCore.ParameterSet.Config as cms
import copy

from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *
from RecoJets.JetAssociationProducers.ic5PFJetTracksAssociatorAtVertex_cfi import *
#PFTauTagInfo Producer
from RecoTauTag.RecoTau.PFRecoTauTagInfoProducer_cfi import *
from RecoTauTag.RecoTau.PFRecoTauProducer_cfi import *

#Discriminators
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByIsolation_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByLeadingTrackFinding_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByLeadingTrackPtCut_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByTrackIsolation_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByECALIsolation_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstElectron_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstMuon_cfi import *

#Discriminators using leading Pion
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByIsolationUsingLeadingPion_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByLeadingPionPtCut_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByTrackIsolationUsingLeadingPion_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByECALIsolationUsingLeadingPion_cfi import *



#HighEfficiency collections
from RecoTauTag.RecoTau.PFRecoTauHighEfficiency_cff import *


PFTauDiscrimination = cms.Sequence(
    pfRecoTauDiscriminationByIsolation*
    pfRecoTauDiscriminationByIsolationUsingLeadingPion*
    pfRecoTauDiscriminationByLeadingTrackFinding*
    pfRecoTauDiscriminationByLeadingTrackPtCut*
    pfRecoTauDiscriminationByLeadingPionPtCut*
    pfRecoTauDiscriminationByTrackIsolation*
    pfRecoTauDiscriminationByTrackIsolationUsingLeadingPion*
    pfRecoTauDiscriminationByECALIsolation*
    pfRecoTauDiscriminationByECALIsolationUsingLeadingPion*
    pfRecoTauDiscriminationAgainstElectron*
    pfRecoTauDiscriminationAgainstMuon
)


PFTauDiscriminationHighEfficiency = cms.Sequence(
    pfRecoTauDiscriminationByIsolationHighEfficiency*
    pfRecoTauDiscriminationByIsolationUsingLeadingPionHighEfficiency*
    pfRecoTauDiscriminationByLeadingTrackFindingHighEfficiency*
    pfRecoTauDiscriminationByLeadingTrackPtCutHighEfficiency*
    pfRecoTauDiscriminationByLeadingPionPtCutHighEfficiency*
    pfRecoTauDiscriminationByTrackIsolationHighEfficiency*
    pfRecoTauDiscriminationByTrackIsolationUsingLeadingPionHighEfficiency*
    pfRecoTauDiscriminationByECALIsolationHighEfficiency*
    pfRecoTauDiscriminationByECALIsolationUsingLeadingPionHighEfficiency*
    pfRecoTauDiscriminationAgainstElectronHighEfficiency*
    pfRecoTauDiscriminationAgainstMuonHighEfficiency
)

PFTauHighEfficiency = cms.Sequence(
    pfRecoTauProducerHighEfficiency*
    PFTauDiscriminationHighEfficiency
)

PFTau = cms.Sequence(
    ic5PFJetTracksAssociatorAtVertex*
    pfRecoTauTagInfoProducer*
    pfRecoTauProducer*
    PFTauDiscrimination*
    PFTauHighEfficiency*
    PFTauDiscriminationHighEfficiency	
)



