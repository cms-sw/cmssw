import FWCore.ParameterSet.Config as cms

from RecoTauTag.Configuration.HPSPFTaus_cff import hpsPFTauDiscriminationByLooseMuonRejection3, hpsPFTauDiscriminationByTightMuonRejection3

hpsPFTauDiscriminationByLooseMuonRejectionSimple = cms.EDProducer("PFRecoTauDiscriminationAgainstMuonSimple",
    PFTauProducer = cms.InputTag("hpsPFTauProducer"),
    Prediscriminants = hpsPFTauDiscriminationByLooseMuonRejection3.Prediscriminants,
    HoPMin = cms.double(0.1), #use smaller value that with AOD as raw energy is used
    doCaloMuonVeto = cms.bool(False), #do not use it until tuned
    srcPatMuons = cms.InputTag("slimmedMuons"),
    minPtMatchedMuon = hpsPFTauDiscriminationByLooseMuonRejection3.minPtMatchedMuon,
    dRmuonMatch = hpsPFTauDiscriminationByLooseMuonRejection3.dRmuonMatch,
    dRmuonMatchLimitedToJetArea = hpsPFTauDiscriminationByLooseMuonRejection3.dRmuonMatchLimitedToJetArea,
    maskHitsCSC = hpsPFTauDiscriminationByLooseMuonRejection3.maskHitsCSC,
    maskHitsDT = hpsPFTauDiscriminationByLooseMuonRejection3.maskHitsDT,
    maskHitsRPC = hpsPFTauDiscriminationByLooseMuonRejection3.maskHitsRPC,
    maxNumberOfHitsLast2Stations = hpsPFTauDiscriminationByLooseMuonRejection3.maxNumberOfHitsLast2Stations,
    maskMatchesCSC = hpsPFTauDiscriminationByLooseMuonRejection3.maskMatchesCSC,
    maskMatchesDT = hpsPFTauDiscriminationByLooseMuonRejection3.maskMatchesDT,
    maskMatchesRPC = hpsPFTauDiscriminationByLooseMuonRejection3.maskMatchesRPC,
    maxNumberOfMatches = hpsPFTauDiscriminationByLooseMuonRejection3.maxNumberOfMatches,
    maxNumberOfSTAMuons = cms.int32(-1),
    maxNumberOfRPCMuons = cms.int32(-1),
    verbosity = cms.int32(0)
)

hpsPFTauDiscriminationByTightMuonRejectionSimple = hpsPFTauDiscriminationByLooseMuonRejectionSimple.clone(
    maxNumberOfHitsLast2Stations = hpsPFTauDiscriminationByTightMuonRejection3.maxNumberOfHitsLast2Stations
)
