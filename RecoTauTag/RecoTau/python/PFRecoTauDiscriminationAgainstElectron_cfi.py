import FWCore.ParameterSet.Config as cms

pfRecoTauDiscriminationAgainstElectron = cms.EDFilter("PFRecoTauDiscriminationAgainstElectron",
    EcalStripSumEOverPLead_maxValue = cms.double(1.8), ## Upper cut away window threshold

    EcalStripSumEOverPLead_minValue = cms.double(0.8), ## Lower cut away window threshold

    ElecPreID0_SumEOverPLead_maxValue = cms.double(0.95),
    ApplyCut_EmFraction = cms.bool(False),
    EmFraction_maxValue = cms.double(0.9),
    PFTauProducer = cms.string('pfRecoTauProducer'),
    HcalTotOverPLead_minValue = cms.double(0.1),
    ApplyCut_ElectronPreID = cms.bool(True),
    BremsRecoveryEOverPLead_maxValue = cms.double(1.8), ## Upper cut away window threshold

    ElecPreID0_Hcal3x3_minValue = cms.double(0.05),
    ApplyCut_BremsRecoveryEOverPLead = cms.bool(False),
    ApplyCut_HcalTotOverPLead = cms.bool(False),
    ApplyCut_HcalMaxOverPLead = cms.bool(False),
    ApplyCut_Hcal3x3OverPLead = cms.bool(False),
    ElecPreID1_Hcal3x3_minValue = cms.double(0.15),
    ElecPreID1_SumEOverPLead_maxValue = cms.double(0.8),
    ApplyCut_EcalCrack_ = cms.bool(True),
    ApplyCut_EcalStripSumEOverPLead = cms.bool(False),
    BremsRecoveryEOverPLead_minValue = cms.double(0.8), ## Lower cut away window threshold

    Hcal3x3OverPLead_minValue = cms.double(0.1),
    HcalMaxOverPLead_minValue = cms.double(0.1)
)


