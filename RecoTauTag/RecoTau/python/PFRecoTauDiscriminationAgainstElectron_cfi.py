import FWCore.ParameterSet.Config as cms

pfRecoTauDiscriminationAgainstElectron = cms.EDFilter("PFRecoTauDiscriminationAgainstElectron",

    PFTauProducer = cms.string('pfRecoTauProducer'),

    ApplyCut_EmFraction = cms.bool(False),
    EmFraction_maxValue = cms.double(0.9),

    ApplyCut_HcalTotOverPLead = cms.bool(False),
    HcalTotOverPLead_minValue = cms.double(0.1),

    ApplyCut_Hcal3x3OverPLead = cms.bool(False),
    Hcal3x3OverPLead_minValue = cms.double(0.1),

    ApplyCut_HcalMaxOverPLead = cms.bool(False),
    HcalMaxOverPLead_minValue = cms.double(0.1)

    ApplyCut_EcalStripSumEOverPLead = cms.bool(False),
    EcalStripSumEOverPLead_maxValue = cms.double(1.8), ## Upper cut away window threshold#
    EcalStripSumEOverPLead_minValue = cms.double(0.8), ## Lower cut away window threshold

    ApplyCut_BremsRecoveryEOverPLead = cms.bool(False),
    BremsRecoveryEOverPLead_minValue = cms.double(0.8), ## Lower cut away window threshold
    BremsRecoveryEOverPLead_maxValue = cms.double(1.8),  ##Upper cut away window threshold

    ApplyCut_ElectronPreID = cms.bool(False), # Electron PreID only

    ApplyCut_ElectronPreID_2D = cms.bool(True),
    ElecPreID0_SumEOverPLead_maxValue = cms.double(0.95),
    ElecPreID0_Hcal3x3_minValue = cms.double(0.05),
    ElecPreID1_SumEOverPLead_maxValue = cms.double(0.8),
   ElecPreID1_Hcal3x3_minValue = cms.double(0.15),
 
    ApplyCut_EcalCrackCut = cms.bool(True),

)


