import FWCore.ParameterSet.Config as cms
import copy

#Necessary for building PFTauTagInfos
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

#Construction of PFTauDecayModes
from RecoTauTag.RecoTau.PFRecoTauDecayModeDeterminator_cfi import *
#HighEfficiency collections
from RecoTauTag.RecoTau.PFRecoTauHighEfficiency_cff import *
#InsideOut collections (Optional)
from RecoTauTag.RecoTau.PFRecoTauInsideOut_cff import *

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

PFTauDiscriminationInsideOut = cms.Sequence(
    pfRecoTauDiscriminationByIsolationInsideOut*
    pfRecoTauDiscriminationByIsolationUsingLeadingPionInsideOut*
    pfRecoTauDiscriminationByLeadingTrackFindingInsideOut*
    pfRecoTauDiscriminationByLeadingTrackPtCutInsideOut*
    pfRecoTauDiscriminationByLeadingPionPtCutInsideOut*
    pfRecoTauDiscriminationByTrackIsolationInsideOut*
    pfRecoTauDiscriminationByTrackIsolationUsingLeadingPionInsideOut*
    pfRecoTauDiscriminationByECALIsolationInsideOut*
    pfRecoTauDiscriminationByECALIsolationUsingLeadingPionInsideOut*
    pfRecoTauDiscriminationAgainstElectronInsideOut*
    pfRecoTauDiscriminationAgainstMuonInsideOut
)

# Produce and discriminate on Inside Out PFTaus (not included in default 
# PFTau sequence)
PFTauInsideOut = cms.Sequence(
    pfRecoTauProducerInsideOut*
    PFTauDiscriminationInsideOut*
    pfTauDecayModeInsideOut
)

# Produce and discriminate on High Efficiency PFTaus
PFTauHighEfficiency = cms.Sequence(
    pfRecoTauProducerHighEfficiency*
    PFTauDiscriminationHighEfficiency*
    pfTauDecayModeHighEfficiency
)

PFTau = cms.Sequence(
    ic5PFJetTracksAssociatorAtVertex*
    pfRecoTauTagInfoProducer*
    pfRecoTauProducer*
    PFTauDiscrimination*
    pfTauDecayMode*
    PFTauHighEfficiency
)



