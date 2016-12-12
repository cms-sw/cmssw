import FWCore.ParameterSet.Config as cms

from RecoTauTag.RecoTau.PFRecoTauQualityCuts_cfi import PFTauQualityCuts
from RecoTauTag.RecoTau.TauDiscriminatorTools import requireLeadTrack

pfRecoTauDiscriminationByIsolation = cms.EDProducer("PFRecoTauDiscriminationByIsolation",
    PFTauProducer = cms.InputTag('pfRecoTauProducer'), #tau collection to discriminate

    # Require leading pion ensures that:
    # 1) these is at least one track above threshold (0.5 GeV) in the signal cone
    # 2) a track in the signal cone has pT > 5 GeV
    Prediscriminants = requireLeadTrack,

    # Select which collections to use for isolation. You can select one or both
    ApplyDiscriminationByECALIsolation = cms.bool(True), # use PFGammas when isolating
    ApplyDiscriminationByTrackerIsolation = cms.bool(True), # use PFChargedHadr when isolating
    ApplyDiscriminationByWeightedECALIsolation = cms.bool(False), #do not use pileup weighting of neutral deposits by default

    applyOccupancyCut = cms.bool(True), # apply a cut on number of isolation objects
    maximumOccupancy = cms.uint32(0), # no tracks > 1 GeV or gammas > 1.5 GeV allowed

    applySumPtCut = cms.bool(False), # apply a cut on the sum Pt of the isolation objects
    maximumSumPtCut = cms.double(6.0),

    applyRelativeSumPtCut = cms.bool(False), # apply a cut on IsoPt/TotalPt
    relativeSumPtCut = cms.double(0.0),
    relativeSumPtOffset = cms.double(0.0),

    minTauPtForNoIso = cms.double(-99.), # minimum tau pt at which the isolation is completely relaxed. If negative, this is disabled
    
    applyPhotonPtSumOutsideSignalConeCut = cms.bool(False),
    maxAbsPhotonSumPt_outsideSignalCone = cms.double(1.e+9),
    maxRelPhotonSumPt_outsideSignalCone = cms.double(0.10),

    qualityCuts = PFTauQualityCuts, # set the standard quality cuts

    # Delta-Beta corrections to remove Pileup
    applyDeltaBetaCorrection = cms.bool(False),
    particleFlowSrc = cms.InputTag("particleFlow"),
    vertexSrc = PFTauQualityCuts.primaryVertexSrc,
    # This must correspond to the cone size of the algorithm which built the
    # tau. (or if customOuterCone option is used, the custom cone size)
    isoConeSizeForDeltaBeta = cms.double(0.5),
    # The delta beta factor maps the expected neutral contribution in the
    # isolation cone from the observed PU charged contribution.  This factor can
    # optionally be a function (use 'x') of the number of vertices in the event
    # (taken from the multiplicity of vertexSrc collection)
    deltaBetaFactor = cms.string("0.38"),
    # By default, the pt threshold for tracks used to compute the DeltaBeta
    # correction is taken as the gamma Et threshold from the isolation quality
    # cuts.
    # Uncommenting the parameter below allows this threshold to be overridden.
    #deltaBetaPUTrackPtCutOverride = cms.double(1.5),

    # Tau footprint correction
    applyFootprintCorrection = cms.bool(False),
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
    applyRhoCorrection = cms.bool(False),
    rhoProducer = cms.InputTag("fixedGridRhoFastjetAll"),
    rhoConeSize = cms.double(0.5),
    rhoUEOffsetCorrection = cms.double(1.0),
    UseAllPFCandsForWeights = cms.bool(False),
    verbosity = cms.int32(0)
                                                   
)
