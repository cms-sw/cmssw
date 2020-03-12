import FWCore.ParameterSet.Config as cms

from RecoTauTag.RecoTau.HLTPFRecoTauQualityCuts_cfi import hltPFTauQualityCuts
from RecoTauTag.RecoTau.TauDiscriminatorTools import requireLeadTrack, noPrediscriminants
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByIsolation_cfi import pfRecoTauDiscriminationByIsolation

hltPFRecoTauDiscriminationByIsolation = pfRecoTauDiscriminationByIsolation.clone(
    PFTauProducer = cms.InputTag('hltPFRecoTauProducer'), #tau collection to discriminate

    # Require leading pion ensures that:
    # 1) these is at least one track above threshold (0.5 GeV) in the signal cone
    # 2) a track in the signal cone has pT > 5 GeV
    Prediscriminants = noPrediscriminants,

    # Select which collections to use for isolation. You can select one or both
    ApplyDiscriminationByECALIsolation = cms.bool(False), # use PFGammas when isolating
    ApplyDiscriminationByTrackerIsolation = cms.bool(True), # use PFChargedHadr when isolating

    applyOccupancyCut = cms.bool(True), # apply a cut on number of isolation objects
    maximumOccupancy = cms.uint32(0), # no tracks > 1 GeV or gammas > 1.5 GeV allowed

    applySumPtCut = cms.bool(False), # apply a cut on the sum Pt of the isolation objects
    maximumSumPtCut = cms.double(6.0),
    storeRawSumPt = cms.bool(False),

    applyRelativeSumPtCut = cms.bool(False), # apply a cut on IsoPt/TotalPt
    relativeSumPtCut = cms.double(0.0),

    qualityCuts = hltPFTauQualityCuts,# set the standard quality cuts

    # Delta-Beta corrections to remove Pileup
    applyDeltaBetaCorrection = cms.bool(False),
    particleFlowSrc = cms.InputTag("hltParticleFlow"),
    vertexSrc = hltPFTauQualityCuts.primaryVertexSrc,
    customOuterCone = cms.double( -1.0 ),

    # This must correspond to the cone size of the algorithm which built the
    # tau. (or if customOuterCone option is used, the custom cone size)
    isoConeSizeForDeltaBeta = cms.double(0.3),
    # The delta beta factor maps the expected neutral contribution in the
    # isolation cone from the observed PU charged contribution.  This factor can
    # optionally be a function (use 'x') of the number of vertices in the event
    # (taken from the multiplicity of vertexSrc collection)
    deltaBetaFactor = cms.string("0.38"),
    # By default, the pt threshold for tracks used to compute the DeltaBeta
    # correction is taken as the gamma Et threshold from the isolation quality
    # cuts.
    deltaBetaPUTrackPtCutOverride     = cms.bool(True),  # Set the boolean = True to override.
    deltaBetaPUTrackPtCutOverride_val = cms.double(0.5), # Set the value for new value.

    # Rho corrections
    applyRhoCorrection = cms.bool(False),
    rhoProducer = cms.InputTag("kt6PFJets", "rho"),
    rhoConeSize = cms.double(0.5),
    rhoUEOffsetCorrection = cms.double(1.0),
)
