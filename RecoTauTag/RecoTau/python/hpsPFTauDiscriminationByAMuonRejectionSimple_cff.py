import FWCore.ParameterSet.Config as cms

from RecoTauTag.Configuration.HPSPFTaus_cff import hpsPFTauDiscriminationByLooseMuonRejection3, hpsPFTauDiscriminationByTightMuonRejection3

hpsPFTauDiscriminationByLooseMuonRejectionSimple = hpsPFTauDiscriminationByLooseMuonRejection3.clone(
    PFTauProducer=cms.InputTag("hpsPFTauProducer"),
    HoPMin=cms.double(0.1), #use smaller value than with AOD as raw energy is used
    doCaloMuonVeto=cms.bool(False), #do not use it until tuned
    srcPatMuons=cms.InputTag("slimmedMuons"),
    maxNumberOfSTAMuons=cms.int32(-1),
    maxNumberOfRPCMuons=cms.int32(-1)
    )

hpsPFTauDiscriminationByLooseMuonRejectionSimple.__dict__['_TypedParameterizable__type'] = "PFRecoTauDiscriminationAgainstMuonSimple"

for attr in ['discriminatorOption', 'srcMuons']:
    delattr(hpsPFTauDiscriminationByLooseMuonRejectionSimple, attr)


hpsPFTauDiscriminationByTightMuonRejectionSimple = hpsPFTauDiscriminationByLooseMuonRejectionSimple.clone(
    maxNumberOfHitsLast2Stations=hpsPFTauDiscriminationByTightMuonRejection3.maxNumberOfHitsLast2Stations
)
