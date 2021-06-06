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
    electrons = "mergedGsfElectronsForTauId"
)
# anti-e phase-2 tauID (raw)
from RecoTauTag.RecoTau.tauDiscriminationAgainstElectronMVA6Phase2_mvaDefs_cff import mvaNames_phase2, mapping_phase2, workingPoints_phase2
from RecoTauTag.RecoTau.pfRecoTauDiscriminationAgainstElectronMVA6_cfi import *
pfRecoTauDiscriminationAgainstElectronMVA6Phase2Raw = pfRecoTauDiscriminationAgainstElectronMVA6.clone(
    #Note: PFTauProducer and Prediscriminants have to be set in the final cfg
    srcElectrons = "mergedGsfElectronsForTauId",
    isPhase2 = True,
    vetoEcalCracks = False,
    hgcalElectronIDs = [cms.InputTag("hgcElectronIdForTauId", key) for key in hgcElectronIdForTauId.variables],
    **mvaNames_phase2
)
# anti-e phase-2 tauID (WPs)
from RecoTauTag.RecoTau.recoTauDiscriminantCutMultiplexerDefault_cfi import recoTauDiscriminantCutMultiplexerDefault
pfRecoTauDiscriminationAgainstElectronMVA6Phase2 = recoTauDiscriminantCutMultiplexerDefault.clone(
    #Note: PFTauProducer and Prediscriminants have to be set in the final cfg
    toMultiplex = "pfRecoTauDiscriminationAgainstElectronMVA6Phase2Raw",
    mapping = mapping_phase2,
    rawValues = ["discriminator", "category"],
    workingPoints = workingPoints_phase2
)

electronsForTauDiscriminationAgainstElectronMVA6Phase2Task = cms.Task(
    cleanedEcalDrivenGsfElectronsFromMultiCl,
    cleanedEcalDrivenGsfElectronsFromMultiClNoEB,
    mergedGsfElectronsForTauId,
    hgcElectronIdForTauId
)

pfRecoTauDiscriminationAgainstElectronMVA6Phase2Task = cms.Task(
    electronsForTauDiscriminationAgainstElectronMVA6Phase2Task,
    pfRecoTauDiscriminationAgainstElectronMVA6Phase2Raw,
    pfRecoTauDiscriminationAgainstElectronMVA6Phase2
)

pfRecoTauDiscriminationAgainstElectronMVA6Phase2Seq = cms.Sequence(
    pfRecoTauDiscriminationAgainstElectronMVA6Phase2Task
)
