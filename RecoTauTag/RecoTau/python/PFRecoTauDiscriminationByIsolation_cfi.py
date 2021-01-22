import FWCore.ParameterSet.Config as cms

from RecoTauTag.RecoTau.PFRecoTauQualityCuts_cfi import PFTauQualityCuts
from RecoTauTag.RecoTau.TauDiscriminatorTools import requireLeadTrack
from RecoTauTag.RecoTau.pfRecoTauDiscriminationByIsolationContainer_cfi import pfRecoTauDiscriminationByIsolationContainer

pfRecoTauDiscriminationByIsolation = pfRecoTauDiscriminationByIsolationContainer.clone(
    # Require leading pion ensures that:
    # 1) these is at least one track above threshold (0.5 GeV) in the signal cone
    # 2) a track in the signal cone has pT > 5 GeV
    Prediscriminants = requireLeadTrack,

    # Select which collections to use for isolation. You can select one or both
    WeightECALIsolation = 1., # apply a flat, overall weight to ECAL isolation. Useful to combine charged and neutral isolations with different relative weights. Default 1. 

    minTauPtForNoIso = -99., # minimum tau pt at which the isolation is completely relaxed. If negative, this is disabled

    qualityCuts = PFTauQualityCuts, # set the standard quality cuts

    # Delta-Beta corrections to remove Pileup
    particleFlowSrc = "particleFlow",
    vertexSrc = PFTauQualityCuts.primaryVertexSrc,
    # This must correspond to the cone size of the algorithm which built the
    # tau. (or if customOuterCone option is used, the custom cone size)
    customOuterCone = -1., # propagated this default from .cc, it probably corresponds to not using customOuterCone
    isoConeSizeForDeltaBeta = 0.5,
    # The delta beta factor maps the expected neutral contribution in the
    # isolation cone from the observed PU charged contribution.  This factor can
    # optionally be a function (use 'x') of the number of vertices in the event
    # (taken from the multiplicity of vertexSrc collection)
    deltaBetaFactor = "0.38",
    # By default, the pt threshold for tracks used to compute the DeltaBeta
    # correction is taken as the gamma Et threshold from the isolation quality
    # cuts.
    deltaBetaPUTrackPtCutOverride     = False,  # Set the boolean = True to override.
    deltaBetaPUTrackPtCutOverride_val = -1.5, # Set the value for new value.

    # Tau footprint correction
    applyFootprintCorrection = False,
    footprintCorrections = cms.VPSet(
        cms.PSet(
            selection = cms.string("decayMode() = 0"),
            offset = cms.string("0.0")
        ),
        cms.PSet(
            selection = cms.string("decayMode() = 1 || decayMode() = 2"),
            offset = cms.string("0.0")
        ),
        cms.PSet(
            selection = cms.string("decayMode() = 5"),
            offset = cms.string("2.7")
        ),
        cms.PSet(
            selection = cms.string("decayMode() = 6"),
            offset = cms.string("0.0")
        ),
        cms.PSet(
            selection = cms.string("decayMode() = 10"),
            offset = cms.string("max(2.0, 0.22*pt() - 2.0)")
        )        
    ),                                                        

    # Rho corrections
    applyRhoCorrection = False,
    rhoProducer = "fixedGridRhoFastjetAll",
    rhoConeSize = 0.5,
    rhoUEOffsetCorrection = 1.0,

    verbosity = 0,
)
