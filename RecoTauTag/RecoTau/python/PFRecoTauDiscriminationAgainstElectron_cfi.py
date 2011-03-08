import FWCore.ParameterSet.Config as cms
from RecoTauTag.RecoTau.TauDiscriminatorTools import requireLeadTrack

pfRecoTauDiscriminationAgainstElectron = cms.EDProducer("PFRecoTauDiscriminationAgainstElectron",

    # tau collection to discriminate
    PFTauProducer = cms.InputTag('pfRecoTauProducer'),

    # Require leading pion ensures that:
    #  1) these is at least one track above threshold (0.5 GeV) in the signal cone
    #  2) a track OR a pi-zero in the signal cone has pT > 5 GeV
    Prediscriminants = requireLeadTrack,

    ApplyCut_EmFraction = cms.bool(False),
    EmFraction_maxValue = cms.double(0.9),

    ApplyCut_HcalTotOverPLead = cms.bool(False),
    HcalTotOverPLead_minValue = cms.double(0.1),

    ApplyCut_Hcal3x3OverPLead = cms.bool(False),
    Hcal3x3OverPLead_minValue = cms.double(0.1),

    ApplyCut_HcalMaxOverPLead = cms.bool(False),
    HcalMaxOverPLead_minValue = cms.double(0.1),

    ApplyCut_EOverPLead = cms.bool(False),
    EOverPLead_maxValue = cms.double(1.8), ## Upper cut away window threshold#
    EOverPLead_minValue = cms.double(0.8), ## Lower cut away window threshold

    ApplyCut_BremsRecoveryEOverPLead = cms.bool(False),
    BremsRecoveryEOverPLead_minValue = cms.double(0.8), ## Lower cut away window threshold
    BremsRecoveryEOverPLead_maxValue = cms.double(1.8),  ##Upper cut away window threshold

    ApplyCut_ElectronPreID = cms.bool(False), # Electron PreID only

    ApplyCut_ElectronPreID_2D = cms.bool(False),
    ElecPreID0_EOverPLead_maxValue = cms.double(0.95),
    ElecPreID0_HOverPLead_minValue = cms.double(0.05),
    ElecPreID1_EOverPLead_maxValue = cms.double(0.8),
    ElecPreID1_HOverPLead_minValue = cms.double(0.15),
 
    ApplyCut_PFElectronMVA = cms.bool(True),
    PFElectronMVA_maxValue = cms.double(-0.1),

    ApplyCut_EcalCrackCut = cms.bool(False),

   ApplyCut_BremCombined = cms.bool(False),
   BremCombined_Fraction  = cms.double(0.99),
   BremCombined_HOP       = cms.double(0.1),
   BremCombined_Mass      = cms.double(0.55),
   BremCombined_StripSize = cms.double(0.03)


)


