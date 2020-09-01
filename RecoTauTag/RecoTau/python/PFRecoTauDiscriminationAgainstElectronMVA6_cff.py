import FWCore.ParameterSet.Config as cms

# HGCal electron stuff
from RecoEgamma.EgammaTools.cleanedEcalDrivenGsfElectronsFromMultiCl_cfi import cleanedEcalDrivenGsfElectronsFromMultiCl
from RecoEgamma.EgammaTools.hgcalElectronIDValueMap_cff import hgcalElectronIDValueMap
# HGCal electrons cleaned against duplicates and electrons in barrel (pt>10GeV)
# TauValElectronSelector defined Validation/RecoTau/plugins/Selectors.cc;
# is there a more intuitive place where such a selector is defined?
cleanedEcalDrivenGsfElectronsFromMultiClNoEB = cms.EDFilter('TauValElectronSelector',
    cut = cms.string('!isEB && pt >= 10.'),
    src = cms.InputTag('cleanedEcalDrivenGsfElectronsFromMultiCl')
)
# Electron collection merger
mergedGsfElectronsForTauId = cms.EDProducer('GsfElectronCollectionMerger',
    src = cms.VInputTag('gedGsfElectrons', 'cleanedEcalDrivenGsfElectronsFromMultiClNoEB')
)
# HGCal EleID with merged electron collection
hgcElectronIdForTauId = hgcalElectronIDValueMap.clone(
    electrons = cms.InputTag("mergedGsfElectronsForTauId"),
)
# anti-e phase-2 tauID
from RecoTauTag.RecoTau.pfRecoTauDiscriminationAgainstElectronMVA6_cfi import *
pfRecoTauDiscriminationAgainstElectronMVA6Phase2 = pfRecoTauDiscriminationAgainstElectronMVA6.clone(
    #Note: PFTauProducer and Prediscriminants have to be set in the final cfg
    srcElectrons = 'mergedGsfElectronsForTauId',
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

electronsForTauDiscriminationAgainstElectronMVA6Phase2Task = cms.Task(
    cleanedEcalDrivenGsfElectronsFromMultiCl,
    cleanedEcalDrivenGsfElectronsFromMultiClNoEB,
    mergedGsfElectronsForTauId,
    hgcElectronIdForTauId
)

pfRecoTauDiscriminationAgainstElectronMVA6Phase2Task = cms.Task(
    electronsForTauDiscriminationAgainstElectronMVA6Phase2Task,
    pfRecoTauDiscriminationAgainstElectronMVA6Phase2
)

pfRecoTauDiscriminationAgainstElectronMVA6Phase2Seq = cms.Sequence(
    pfRecoTauDiscriminationAgainstElectronMVA6Phase2Task
)
