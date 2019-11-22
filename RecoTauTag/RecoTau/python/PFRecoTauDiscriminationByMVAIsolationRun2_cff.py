import FWCore.ParameterSet.Config as cms

from RecoTauTag.RecoTau.PFRecoTauDiscriminationByIsolation_cfi import *
from RecoTauTag.RecoTau.RecoTauDiscriminantCutMultiplexer_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByMVAIsolation2_cff import *

#chargedIsoPtSum = pfRecoTauDiscriminationByIsolation.clone(
#    PFTauProducer = cms.InputTag('pfTauProducer'),
#    ApplyDiscriminationByECALIsolation = cms.bool(False),
#    ApplyDiscriminationByTrackerIsolation = cms.bool(True),
#    applyOccupancyCut = cms.bool(False),
#    applySumPtCut = cms.bool(False),
#    applyDeltaBetaCorrection = cms.bool(False),
#    storeRawSumPt = cms.bool(True),
#    storeRawPUsumPt = cms.bool(False),
#    customOuterCone = cms.double(0.5),
#    isoConeSizeForDeltaBeta = cms.double(0.8),
#    verbosity = cms.int32(0)
#)
#neutralIsoPtSum = pfRecoTauDiscriminationByIsolation.clone(
#    PFTauProducer = cms.InputTag('pfTauProducer'),
#    ApplyDiscriminationByECALIsolation = cms.bool(True),
#    ApplyDiscriminationByTrackerIsolation = cms.bool(False),
#    applyOccupancyCut = cms.bool(False),
#    applySumPtCut = cms.bool(False),
#    applyDeltaBetaCorrection = cms.bool(False),
#    storeRawSumPt = cms.bool(True),
#    storeRawPUsumPt = cms.bool(False),
#    customOuterCone = cms.double(0.5),
#    isoConeSizeForDeltaBeta = cms.double(0.8),
#    verbosity = cms.int32(0)
#)
#puCorrPtSum = pfRecoTauDiscriminationByIsolation.clone(
#    PFTauProducer = cms.InputTag('pfTauProducer'),
#    ApplyDiscriminationByECALIsolation = cms.bool(False),
#    ApplyDiscriminationByTrackerIsolation = cms.bool(False),
#    applyOccupancyCut = cms.bool(False),
#    applySumPtCut = cms.bool(False),
#    applyDeltaBetaCorrection = cms.bool(True),
#    storeRawSumPt = cms.bool(False),
#    storeRawPUsumPt = cms.bool(True),
#    customOuterCone = cms.double(0.5),
#    isoConeSizeForDeltaBeta = cms.double(0.8),
#    verbosity = cms.int32(0)
#)

photonPtSumOutsideSignalCone = chargedIsoPtSum.clone(
    ApplyDiscriminationByTrackerIsolation = cms.bool(False),
    storeRawSumPt = cms.bool(False),
    storeRawPhotonSumPt_outsideSignalCone = cms.bool(True),
    verbosity = cms.int32(0)
)

footprintCorrection = chargedIsoPtSum.clone(
    ApplyDiscriminationByTrackerIsolation = cms.bool(False),
    storeRawSumPt = cms.bool(False),
    storeRawFootprintCorrection = cms.bool(True),
    verbosity = cms.int32(0)
)

discriminationByIsolationMVArun2v1raw = cms.EDProducer("PFRecoTauDiscriminationByMVAIsolationRun2",

    # tau collection to discriminate
    PFTauProducer = cms.InputTag('pfTauProducer'),

    # Require leading pion ensures that:
    #  1) these is at least one track above threshold (0.5 GeV) in the signal cone
    #  2) a track OR a pi-zero in the signal cone has pT > 5 GeV
    Prediscriminants = requireLeadTrack,
    loadMVAfromDB = cms.bool(True),
    inputFileName = cms.FileInPath("RecoTauTag/RecoTau/data/emptyMVAinputFile"), # the filename for MVA if it is not loaded from DB
    mvaName = cms.string("tauIdMVAnewDMwLT"),
    mvaOpt = cms.string("newDMwLT"),

    # NOTE: tau lifetime reconstruction sequence needs to be run before
    srcTauTransverseImpactParameters = cms.InputTag(''),
    
    srcChargedIsoPtSum = cms.InputTag('chargedIsoPtSum'),
    srcNeutralIsoPtSum = cms.InputTag('neutralIsoPtSum'),
    srcPUcorrPtSum = cms.InputTag('puCorrPtSum'),
    srcPhotonPtSumOutsideSignalCone = cms.InputTag('photonPtSumOutsideSignalCone'),
    srcFootprintCorrection = cms.InputTag('footprintCorrection'),

    verbosity = cms.int32(0)
)

discriminationByIsolationMVArun2v1VLoose = recoTauDiscriminantCutMultiplexer.clone(
    PFTauProducer = cms.InputTag('pfTauProducer'),    
    Prediscriminants = requireLeadTrack,
    toMultiplex = cms.InputTag('discriminationByIsolationMVArun2v1raw'),
    key = cms.InputTag('discriminationByIsolationMVArun2v1raw:category'),
    loadMVAfromDB = cms.bool(True),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0),
            cut = cms.string("newDMwLTEff80"),
            variable = cms.string("pt"),
        )
    )
)
discriminationByIsolationMVArun2v1Loose = discriminationByIsolationMVArun2v1VLoose.clone()
discriminationByIsolationMVArun2v1Loose.mapping[0].cut = cms.string("newDMwLTEff70")
discriminationByIsolationMVArun2v1Medium = discriminationByIsolationMVArun2v1VLoose.clone()
discriminationByIsolationMVArun2v1Medium.mapping[0].cut = cms.string("newDMwLTEff60")
discriminationByIsolationMVArun2v1Tight = discriminationByIsolationMVArun2v1VLoose.clone()
discriminationByIsolationMVArun2v1Tight.mapping[0].cut = cms.string("newDMwLTEff50")
discriminationByIsolationMVArun2v1VTight = discriminationByIsolationMVArun2v1VLoose.clone()
discriminationByIsolationMVArun2v1VTight.mapping[0].cut = cms.string("newDMwLTEff40")

mvaIsolation2TaskRun2 = cms.Task(
    chargedIsoPtSum
   , neutralIsoPtSum
   , puCorrPtSum
   , photonPtSumOutsideSignalCone
   , footprintCorrection
   , discriminationByIsolationMVArun2v1raw
   , discriminationByIsolationMVArun2v1VLoose
   , discriminationByIsolationMVArun2v1Loose
   , discriminationByIsolationMVArun2v1Medium
   , discriminationByIsolationMVArun2v1Tight
   , discriminationByIsolationMVArun2v1VTight
)
mvaIsolation2SeqRun2 = cms.Sequence(mvaIsolation2TaskRun2)
