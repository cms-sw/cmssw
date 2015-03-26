import FWCore.ParameterSet.Config as cms

from RecoTauTag.RecoTau.PFRecoTauDiscriminationByIsolation_cfi import *
from RecoTauTag.RecoTau.RecoTauDiscriminantCutMultiplexer_cfi import *

chargedIsoPtSum = pfRecoTauDiscriminationByIsolation.clone(
    PFTauProducer = cms.InputTag('pfTauProducer'),
    ApplyDiscriminationByECALIsolation = cms.bool(False),
    ApplyDiscriminationByTrackerIsolation = cms.bool(True),
    applyOccupancyCut = cms.bool(False),
    applySumPtCut = cms.bool(False),
    applyDeltaBetaCorrection = cms.bool(False),
    storeRawSumPt = cms.bool(True),
    storeRawPUsumPt = cms.bool(False),
    customOuterCone = cms.double(0.5),
    isoConeSizeForDeltaBeta = cms.double(0.8),
    verbosity = cms.int32(0)
)
neutralIsoPtSum = pfRecoTauDiscriminationByIsolation.clone(
    PFTauProducer = cms.InputTag('pfTauProducer'),
    ApplyDiscriminationByECALIsolation = cms.bool(True),
    ApplyDiscriminationByTrackerIsolation = cms.bool(False),
    applyOccupancyCut = cms.bool(False),
    applySumPtCut = cms.bool(False),
    applyDeltaBetaCorrection = cms.bool(False),
    storeRawSumPt = cms.bool(True),
    storeRawPUsumPt = cms.bool(False),
    customOuterCone = cms.double(0.5),
    isoConeSizeForDeltaBeta = cms.double(0.8),
    verbosity = cms.int32(0)
)
puCorrPtSum = pfRecoTauDiscriminationByIsolation.clone(
    PFTauProducer = cms.InputTag('pfTauProducer'),
    ApplyDiscriminationByECALIsolation = cms.bool(False),
    ApplyDiscriminationByTrackerIsolation = cms.bool(False),
    applyOccupancyCut = cms.bool(False),
    applySumPtCut = cms.bool(False),
    applyDeltaBetaCorrection = cms.bool(True),
    storeRawSumPt = cms.bool(False),
    storeRawPUsumPt = cms.bool(True),
    customOuterCone = cms.double(0.5),
    isoConeSizeForDeltaBeta = cms.double(0.8),
    verbosity = cms.int32(0)
)

discriminationByIsolationMVA2raw = cms.EDProducer("PFRecoTauDiscriminationByIsolationMVA2",

    # tau collection to discriminate
    PFTauProducer = cms.InputTag('pfTauProducer'),

    # Require leading pion ensures that:
    #  1) these is at least one track above threshold (0.5 GeV) in the signal cone
    #  2) a track OR a pi-zero in the signal cone has pT > 5 GeV
    Prediscriminants = requireLeadTrack,
    loadMVAfromDB = cms.bool(True),
    mvaName = cms.string("tauIdMVAnewDMwLT"),
    mvaOpt = cms.string("newDMwLT"),

    # NOTE: tau lifetime reconstruction sequence needs to be run before
    srcTauTransverseImpactParameters = cms.InputTag(''),
    
    srcChargedIsoPtSum = cms.InputTag('chargedIsoPtSum'),
    srcNeutralIsoPtSum = cms.InputTag('neutralIsoPtSum'),
    srcPUcorrPtSum = cms.InputTag('puCorrPtSum')
)

discriminationByIsolationMVA2VLoose = recoTauDiscriminantCutMultiplexer.clone(
    PFTauProducer = cms.InputTag('pfTauProducer'),    
    Prediscriminants = requireLeadTrack,
    toMultiplex = cms.InputTag('discriminationByIsolationMVA2raw'),
    key = cms.InputTag('discriminationByIsolationMVA2raw:category'),
    loadMVAfromDB = cms.bool(True),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0),
            cut = cms.string("newDMwLTEff80"),
            variable = cms.string("pt"),
        )
    )
)
discriminationByIsolationMVA2Loose = discriminationByIsolationMVA2VLoose.clone()
discriminationByIsolationMVA2Loose.mapping[0].cut = cms.string("newDMwLTEff70")
discriminationByIsolationMVA2Medium = discriminationByIsolationMVA2VLoose.clone()
discriminationByIsolationMVA2Medium.mapping[0].cut = cms.string("newDMwLTEff60")
discriminationByIsolationMVA2Tight = discriminationByIsolationMVA2VLoose.clone()
discriminationByIsolationMVA2Tight.mapping[0].cut = cms.string("newDMwLTEff50")
discriminationByIsolationMVA2VTight = discriminationByIsolationMVA2VLoose.clone()
discriminationByIsolationMVA2VTight.mapping[0].cut = cms.string("newDMwLTEff40")

mvaIsolation2Seq = cms.Sequence(
    chargedIsoPtSum
   + neutralIsoPtSum
   + puCorrPtSum
   + discriminationByIsolationMVA2raw
   + discriminationByIsolationMVA2VLoose
   + discriminationByIsolationMVA2Loose
   + discriminationByIsolationMVA2Medium
   + discriminationByIsolationMVA2Tight
   + discriminationByIsolationMVA2VTight
)
