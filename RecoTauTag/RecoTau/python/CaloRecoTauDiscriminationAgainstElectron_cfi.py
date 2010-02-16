import FWCore.ParameterSet.Config as cms
from RecoTauTag.RecoTau.TauDiscriminatorTools import requireLeadTrackCalo

caloRecoTauDiscriminationAgainstElectron = cms.EDProducer("CaloRecoTauDiscriminationAgainstElectron",

    CaloTauProducer = cms.InputTag('caloRecoTauProducer'),

    Prediscriminants = requireLeadTrackCalo,

    ApplyCut_maxleadTrackHCAL3x3hottesthitDEta = cms.bool(False),
    leadTrack_HCAL3x3hitsEtSumOverPt_minvalue  = cms.double(0.1),
    ApplyCut_leadTrackavoidsECALcrack          = cms.bool(True),
    maxleadTrackHCAL3x3hottesthitDEta          = cms.double(0.1)
)


