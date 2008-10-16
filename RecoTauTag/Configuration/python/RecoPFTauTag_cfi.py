import FWCore.ParameterSet.Config as cms

from RecoTauTag.RecoTau.PFRecoTauDiscriminationByIsolation_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByLeadingTrackFinding_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByLeadingTrackPtCut_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByTrackIsolation_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByECALIsolation_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstElectron_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstMuon_cfi import *

from RecoTauTag.RecoTau.PFRecoTauHighEfficiency_cff import *

PFTauDiscrimination = cms.Sequence(
    pfRecoTauDiscriminationByIsolation*
    pfRecoTauDiscriminationByLeadingTrackFinding*
    pfRecoTauDiscriminationByLeadingTrackPtCut*
    pfRecoTauDiscriminationByTrackIsolation*
    pfRecoTauDiscriminationByECALIsolation*
    pfRecoTauDiscriminationAgainstElectron*
    pfRecoTauDiscriminationAgainstMuon
)

PFTauDiscriminationHighEfficiency = cms.Sequence(
    pfRecoTauDiscriminationByIsolationHighEfficiency*
    pfRecoTauDiscriminationByLeadingTrackFindingHighEfficiency*
    pfRecoTauDiscriminationByLeadingTrackPtCutHighEfficiency*
    pfRecoTauDiscriminationByTrackIsolationHighEfficiency*
    pfRecoTauDiscriminationByECALIsolationHighEfficiency*
    pfRecoTauDiscriminationAgainstElectronHighEfficiency*
    pfRecoTauDiscriminationAgainstMuonHighEfficiency
)
