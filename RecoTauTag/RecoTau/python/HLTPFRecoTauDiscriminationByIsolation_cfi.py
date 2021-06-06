import FWCore.ParameterSet.Config as cms

from RecoTauTag.RecoTau.HLTPFRecoTauQualityCuts_cfi import hltPFTauQualityCuts
from RecoTauTag.RecoTau.TauDiscriminatorTools import requireLeadTrack, noPrediscriminants
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByIsolation_cfi import pfRecoTauDiscriminationByIsolation

hltPFRecoTauDiscriminationByIsolation = pfRecoTauDiscriminationByIsolation.clone(
    PFTauProducer = 'hltPFRecoTauProducer', #tau collection to discriminate

    # Require leading pion ensures that:
    # 1) these is at least one track above threshold (0.5 GeV) in the signal cone
    # 2) a track in the signal cone has pT > 5 GeV
    Prediscriminants = noPrediscriminants,

    qualityCuts = hltPFTauQualityCuts,# set the standard quality cuts

    # Delta-Beta corrections to remove Pileup
    particleFlowSrc = "hltParticleFlow",
    vertexSrc = hltPFTauQualityCuts.primaryVertexSrc,
    customOuterCone = -1.0,

    # This must correspond to the cone size of the algorithm which built the
    # tau. (or if customOuterCone option is used, the custom cone size)
    isoConeSizeForDeltaBeta = 0.3,
    # The delta beta factor maps the expected neutral contribution in the
    # isolation cone from the observed PU charged contribution.  This factor can
    # optionally be a function (use 'x') of the number of vertices in the event
    # (taken from the multiplicity of vertexSrc collection)
    deltaBetaFactor = "0.38",
    # By default, the pt threshold for tracks used to compute the DeltaBeta
    # correction is taken as the gamma Et threshold from the isolation quality
    # cuts.
    deltaBetaPUTrackPtCutOverride     = True,  # Set the boolean = True to override.
    deltaBetaPUTrackPtCutOverride_val = 0.5, # Set the value for new value.

    # Rho corrections
    applyRhoCorrection = False,
    rhoProducer = "kt6PFJets:rho",
    rhoConeSize = 0.5,
    rhoUEOffsetCorrection = 1.0,

    IDdefinitions = cms.VPSet(),
    IDWPdefinitions = cms.VPSet(
        cms.PSet(
            IDname = cms.string("pfRecoTauDiscriminationByIsolation"),
            maximumOccupancy = cms.uint32(0), # no tracks > 1 GeV or gammas > 1.5 GeV allowed
            ApplyDiscriminationByTrackerIsolation = cms.bool(True), # use PFGammas when isolating
        )
    ),
)
