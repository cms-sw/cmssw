import FWCore.ParameterSet.Config as cms

# HGCal electron stuff
from RecoEgamma.EgammaTools.cleanedEcalDrivenGsfElectronsFromMultiCl_cfi import cleanedEcalDrivenGsfElectronsFromMultiCl
from RecoEgamma.EgammaTools.hgcalElectronIDValueMap_cff import hgcalElectronIDValueMap

cleanedEcalDrivenGsfElectronsFromMultiClNoEB = cleanedEcalDrivenGsfElectronsFromMultiCl.clone(
    cleanBarrel = True
)
hgcElectronIDNoEB = hgcalElectronIDValueMap.clone(
    electrons = cms.InputTag("cleanedEcalDrivenGsfElectronsFromMultiClNoEB"),
)

# Electron collection merger
mergedGsfElectronsForTauId = cms.EDProducer('GsfElectronCollectionMerger',
    src = cms.VInputTag('gedGsfElectrons', 'cleanedEcalDrivenGsfElectronsFromMultiClNoEB')
)
# anti-e phase-2 tauID
from RecoTauTag.RecoTau.pfRecoTauDiscriminationAgainstElectronMVA6_cfi import *
pfRecoTauDiscriminationAgainstElectronMVA6Phase2 = pfRecoTauDiscriminationAgainstElectronMVA6.clone(
    #FIXME, correct settings for phase2
    srcElectrons = 'mergedGsfElectronsForTauId'
)

pfRecoTauDiscriminationAgainstElectronMVA6Phase2Task = cms.Task(
    cleanedEcalDrivenGsfElectronsFromMultiClNoEB,
    hgcElectronIDNoEB,
    mergedGsfElectronsForTauId,
    pfRecoTauDiscriminationAgainstElectronMVA6Phase2
)

pfRecoTauDiscriminationAgainstElectronMVA6Phase2Seq = cms.Sequence(
    pfRecoTauDiscriminationAgainstElectronMVA6Phase2Task
)
