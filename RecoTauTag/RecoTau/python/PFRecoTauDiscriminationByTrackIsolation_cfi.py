import FWCore.ParameterSet.Config as cms

from RecoTauTag.RecoTau.PFRecoTauQualityCuts_cfi import PFTauQualityCuts
from RecoTauTag.RecoTau.TauDiscriminatorTools import requireLeadTrack
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByIsolation_cfi import  pfRecoTauDiscriminationByIsolation

pfRecoTauDiscriminationByTrackIsolation = pfRecoTauDiscriminationByIsolation.clone(

    PFTauProducer = cms.InputTag('pfRecoTauProducer'), #tau collection to discriminate

    # Require leading pion ensures that:
    #  1) these is at least one track above threshold (0.5 GeV) in the signal cone
    #  2) a track in the signal cone has pT > 5 GeV
    Prediscriminants = requireLeadTrack,

    # Select which collections to use for isolation.  You can select one or both
    ApplyDiscriminationByECALIsolation    = cms.bool(False),  # use PFGammas when isolating
    ApplyDiscriminationByTrackerIsolation = cms.bool(True),   # use PFChargedHadr when isolating

    applyOccupancyCut                     = cms.bool(True),  # apply a cut on number of isolation objects
    maximumOccupancy                      = cms.uint32(0),   # no tracks > 1 GeV allowed

    applySumPtCut                         = cms.bool(False), # apply a cut on the sum Pt of the isolation objects
    maximumSumPtCut                       = cms.double(6.0),

    applyRelativeSumPtCut                 = cms.bool(False), # apply a cut on IsoPt/TotalPt
    relativeSumPtCut                      = cms.double(0.0),

    qualityCuts                           = PFTauQualityCuts,# set the standard quality cuts
)

