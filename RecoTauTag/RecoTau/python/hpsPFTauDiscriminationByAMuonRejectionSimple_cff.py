import FWCore.ParameterSet.Config as cms

from RecoTauTag.Configuration.HPSPFTaus_cff import hpsPFTauDiscriminationByMuonRejection3

hpsPFTauDiscriminationByLooseMuonRejectionSimple = hpsPFTauDiscriminationByMuonRejection3.clone(
    PFTauProducer = cms.InputTag("hpsPFTauProducer"),
    HoPMin = hpsPFTauDiscriminationByMuonRejection3.IDWPdefinitions[0].HoPMin,
    doCaloMuonVeto = cms.bool(True),
    srcPatMuons = cms.InputTag("slimmedMuons"),
    maxNumberOfSTAMuons = cms.int32(-1),
    maxNumberOfRPCMuons = cms.int32(-1),
    maxNumberOfMatches = hpsPFTauDiscriminationByMuonRejection3.IDWPdefinitions[0].maxNumberOfMatches,
    maxNumberOfHitsLast2Stations = hpsPFTauDiscriminationByMuonRejection3.IDWPdefinitions[0].maxNumberOfHitsLast2Stations
    )

hpsPFTauDiscriminationByLooseMuonRejectionSimple.__dict__['_TypedParameterizable__type'] = "PFRecoTauDiscriminationAgainstMuonSimple"

for attr in ['IDWPdefinitions', 'srcMuons']:
    delattr(hpsPFTauDiscriminationByLooseMuonRejectionSimple, attr)


hpsPFTauDiscriminationByTightMuonRejectionSimple = hpsPFTauDiscriminationByLooseMuonRejectionSimple.clone(
    maxNumberOfHitsLast2Stations = hpsPFTauDiscriminationByMuonRejection3.IDWPdefinitions[1].maxNumberOfHitsLast2Stations
)
