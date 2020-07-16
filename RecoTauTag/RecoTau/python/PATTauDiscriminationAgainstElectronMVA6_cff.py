import FWCore.ParameterSet.Config as cms

# Electron collection merger
mergedSlimmedElectronsForTauId = cms.EDProducer('PATElectronCollectionMerger',
    src = cms.VInputTag('slimmedElectrons', 'slimmedElectronsFromMultiClNoEB')
)
# anti-e phase-2 tauID
from RecoTauTag.RecoTau.patTauDiscriminationAgainstElectronMVA6_cfi import *
patTauDiscriminationAgainstElectronMVA6Phase2 = patTauDiscriminationAgainstElectronMVA6.clone(
    #FIXME, correct settings for phase2
    srcElectrons = 'mergedSlimmedElectronsForTauId'
)

patTauDiscriminationAgainstElectronMVA6Phase2Task = cms.Task(
    mergedSlimmedElectronsForTauId,
    patTauDiscriminationAgainstElectronMVA6Phase2
)

patTauDiscriminationAgainstElectronMVA6Phase2Seq = cms.Sequence(
    patTauDiscriminationAgainstElectronMVA6Phase2Task
)
