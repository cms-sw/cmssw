import FWCore.ParameterSet.Config as cms
from RecoTauTag.RecoTau.TauDiscriminatorTools import requireLeadTrack

pfRecoTauDiscriminationAgainstElectron2 = cms.EDProducer("PFRecoTauDiscriminationAgainstElectron2",

    # tau collection to discriminate
    PFTauProducer = cms.InputTag('pfRecoTauProducer'),

    # Require leading pion ensures that:
    #  1) these is at least one track above threshold (0.5 GeV) in the signal cone
    #  2) a track OR a pi-zero in the signal cone has pT > 5 GeV
    Prediscriminants = requireLeadTrack,

    #cuts to be applied
    keepTausInEcalCrack = cms.bool(True), 
    rejectTausInEcalCrack = cms.bool(False),
    etaCracks = cms.vstring("0.0:0.018","0.423:0.461","0.770:0.806","1.127:1.163","1.460:1.558"),
                                                         
    applyCut_hcal3x3OverPLead = cms.bool(True),
    applyCut_leadPFChargedHadrEoP = cms.bool(True),
    applyCut_GammaEtaMom = cms.bool(False),
    applyCut_GammaPhiMom = cms.bool(False),
    applyCut_GammaEnFrac = cms.bool(True),
    applyCut_HLTSpecific = cms.bool(True),
                                                        
    LeadPFChargedHadrEoP_barrel_min = cms.double(0.99),
    LeadPFChargedHadrEoP_barrel_max = cms.double(1.01),
    Hcal3x3OverPLead_barrel_max = cms.double(0.2),
    GammaEtaMom_barrel_max = cms.double(1.5),
    GammaPhiMom_barrel_max = cms.double(1.5),
    GammaEnFrac_barrel_max = cms.double(0.15),
    LeadPFChargedHadrEoP_endcap_min1 = cms.double(0.7),
    LeadPFChargedHadrEoP_endcap_max1 = cms.double(1.3),
    LeadPFChargedHadrEoP_endcap_min2 = cms.double(0.99),
    LeadPFChargedHadrEoP_endcap_max2 = cms.double(1.01),
    Hcal3x3OverPLead_endcap_max = cms.double(0.1),
    GammaEtaMom_endcap_max = cms.double(1.5),
    GammaPhiMom_endcap_max = cms.double(1.5),
    GammaEnFrac_endcap_max = cms.double(0.2),
    verbosity = cms.int32(0)
)


