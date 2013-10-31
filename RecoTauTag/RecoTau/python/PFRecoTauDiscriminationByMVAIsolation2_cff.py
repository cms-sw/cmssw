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

    inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/gbrDiscriminationByIsolationMVA3_newDMwLT.root'),
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
    inputFileName = cms.FileInPath('RecoTauTag/RecoTau/data/wpDiscriminationByIsolationMVA3_newDMwLT.root'),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0),
            cut = cms.string("newDMwLTeff85"),
            variable = cms.string("pt"),
        )
    )
)
discriminationByIsolationMVA2Loose = discriminationByIsolationMVA2VLoose.clone()
discriminationByIsolationMVA2Loose.mapping[0].cut = cms.string("newDMwLTeff75")
discriminationByIsolationMVA2Medium = discriminationByIsolationMVA2VLoose.clone()
discriminationByIsolationMVA2Medium.mapping[0].cut = cms.string("newDMwLTeff65")
discriminationByIsolationMVA2Tight = discriminationByIsolationMVA2VLoose.clone()
discriminationByIsolationMVA2Tight.mapping[0].cut = cms.string("newDMwLTeff55")
discriminationByIsolationMVA2VTight = discriminationByIsolationMVA2VLoose.clone()
discriminationByIsolationMVA2VTight.mapping[0].cut = cms.string("newDMwLTeff45")

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
