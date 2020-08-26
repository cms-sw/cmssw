import FWCore.ParameterSet.Config as cms

# Electron collection merger
mergedSlimmedElectronsForTauId = cms.EDProducer('PATElectronCollectionMerger',
    src = cms.VInputTag('slimmedElectrons', 'slimmedElectronsFromMultiCl')
)
# anti-e phase-2 tauID
from RecoTauTag.RecoTau.TauDiscriminatorTools import noPrediscriminants
from RecoTauTag.RecoTau.patTauDiscriminationAgainstElectronMVA6_cfi import *
patTauDiscriminationAgainstElectronMVA6Phase2 = patTauDiscriminationAgainstElectronMVA6.clone(
    #Note: PATTauProducer has to be set in the final cfg
    Prediscriminants = noPrediscriminants,
    srcElectrons = 'mergedSlimmedElectronsForTauId',
    isPhase2 = True,
    vetoEcalCracks = False,
    mvaName_NoEleMatch_woGwoGSF_BL = "RecoTauTag_antiElectronPhase2MVA6v1_gbr_NoEleMatch_woGwoGSF_BL",
    mvaName_NoEleMatch_wGwoGSF_BL = "RecoTauTag_antiElectronPhase2MVA6v1_gbr_NoEleMatch_wGwoGSF_BL",
    mvaName_woGwGSF_BL = "RecoTauTag_antiElectronPhase2MVA6v1_gbr_woGwGSF_BL",
    mvaName_wGwGSF_BL = "RecoTauTag_antiElectronPhase2MVA6v1_gbr_wGwGSF_BL",
    mvaName_NoEleMatch_woGwoGSF_EC = "RecoTauTag_antiElectronPhase2MVA6v1_gbr_NoEleMatch_woGwoGSF_FWEC",
    mvaName_NoEleMatch_wGwoGSF_EC = "RecoTauTag_antiElectronPhase2MVA6v1_gbr_NoEleMatch_wGwoGSF_FWEC",
    mvaName_woGwGSF_EC = "RecoTauTag_antiElectronPhase2MVA6v1_gbr_woGwGSF_FWEC",
    mvaName_wGwGSF_EC = "RecoTauTag_antiElectronPhase2MVA6v1_gbr_wGwGSF_FWEC",
    mvaName_NoEleMatch_woGwoGSF_VFEC = "RecoTauTag_antiElectronPhase2MVA6v1_gbr_NoEleMatch_woGwoGSF_VFWEC",
    mvaName_NoEleMatch_wGwoGSF_VFEC = "RecoTauTag_antiElectronPhase2MVA6v1_gbr_NoEleMatch_wGwoGSF_VFWEC",
    mvaName_woGwGSF_VFEC = "RecoTauTag_antiElectronPhase2MVA6v1_gbr_woGwGSF_VFWEC",
    mvaName_wGwGSF_VFEC = "RecoTauTag_antiElectronPhase2MVA6v1_gbr_wGwGSF_VFWEC"
)

patTauDiscriminationAgainstElectronMVA6Phase2Task = cms.Task(
    mergedSlimmedElectronsForTauId,
    patTauDiscriminationAgainstElectronMVA6Phase2
)

patTauDiscriminationAgainstElectronMVA6Phase2Seq = cms.Sequence(
    patTauDiscriminationAgainstElectronMVA6Phase2Task
)
