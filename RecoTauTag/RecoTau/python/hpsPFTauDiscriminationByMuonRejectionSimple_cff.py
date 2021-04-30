import FWCore.ParameterSet.Config as cms
import copy

from RecoTauTag.RecoTau.pfRecoTauDiscriminationAgainstMuonSimple_cfi import pfRecoTauDiscriminationAgainstMuonSimple
from RecoTauTag.Configuration.HPSPFTaus_cff import hpsPFTauDiscriminationByMuonRejection3

IDWPdefinitionsSimple = cms.VPSet()
for wp in hpsPFTauDiscriminationByMuonRejection3.IDWPdefinitions:
    aWP = copy.deepcopy(wp)
    aWP.IDname = wp.IDname.value().replace('MuonRejection3','MuonRejectionSimple')
    del aWP.discriminatorOption
    aWP.maxNumberOfRPCMuons = cms.int32(-1)
    aWP.maxNumberOfSTAMuons = cms.int32(-1)
    IDWPdefinitionsSimple.append(aWP)

hpsPFTauDiscriminationByMuonRejectionSimple = pfRecoTauDiscriminationAgainstMuonSimple.clone(
    PFTauProducer = hpsPFTauDiscriminationByMuonRejection3.PFTauProducer,
    Prediscriminants = hpsPFTauDiscriminationByMuonRejection3.Prediscriminants,
    IDWPdefinitions = IDWPdefinitionsSimple,
    dRmuonMatch = hpsPFTauDiscriminationByMuonRejection3.dRmuonMatch,
    dRmuonMatchLimitedToJetArea = hpsPFTauDiscriminationByMuonRejection3.dRmuonMatchLimitedToJetArea,
    minPtMatchedMuon = hpsPFTauDiscriminationByMuonRejection3.minPtMatchedMuon,
    maskMatchesDT = hpsPFTauDiscriminationByMuonRejection3.maskMatchesDT,
    maskMatchesCSC = hpsPFTauDiscriminationByMuonRejection3.maskMatchesCSC,
    maskMatchesRPC = hpsPFTauDiscriminationByMuonRejection3.maskMatchesRPC,
    maskHitsDT = hpsPFTauDiscriminationByMuonRejection3.maskHitsDT,
    maskHitsCSC = hpsPFTauDiscriminationByMuonRejection3.maskHitsCSC,
    maskHitsRPC = hpsPFTauDiscriminationByMuonRejection3.maskHitsRPC
)
