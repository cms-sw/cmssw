import FWCore.ParameterSet.Config as cms

# Electron collection merger
mergedSlimmedElectronsForTauId = cms.EDProducer('PATElectronCollectionMerger',
    src = cms.VInputTag('slimmedElectrons', 'slimmedElectronsFromMultiCl')
)
# anti-e phase-2 tauID (Raw)
from RecoTauTag.RecoTau.tauDiscriminationAgainstElectronMVA6Phase2_mvaDefs_cff import mvaNames_phase2, mapping_phase2, workingPoints_phase2
from RecoTauTag.RecoTau.TauDiscriminatorTools import noPrediscriminants
from RecoTauTag.RecoTau.patTauDiscriminationAgainstElectronMVA6_cfi import *
patTauDiscriminationAgainstElectronMVA6Phase2Raw = patTauDiscriminationAgainstElectronMVA6.clone(
    #Note: PATTauProducer has to be set in the final cfg
    Prediscriminants = noPrediscriminants,
    srcElectrons = "mergedSlimmedElectronsForTauId",
    isPhase2 = True,
    vetoEcalCracks = False,
    **mvaNames_phase2
)
# anti-e phase-2 tauID (WPs)
from RecoTauTag.RecoTau.patTauDiscriminantCutMultiplexerDefault_cfi import patTauDiscriminantCutMultiplexerDefault
patTauDiscriminationAgainstElectronMVA6Phase2 = patTauDiscriminantCutMultiplexerDefault.clone(
    #Note: PFTauProducer and Prediscriminants have to be set in the final cfg
    toMultiplex = 'patTauDiscriminationAgainstElectronMVA6Phase2Raw',
    mapping = mapping_phase2,
    rawValues = ["discriminator", "category"],
    workingPoints = workingPoints_phase2
)

patTauDiscriminationAgainstElectronMVA6Phase2Task = cms.Task(
    mergedSlimmedElectronsForTauId,
    patTauDiscriminationAgainstElectronMVA6Phase2Raw,
    patTauDiscriminationAgainstElectronMVA6Phase2
)

patTauDiscriminationAgainstElectronMVA6Phase2Seq = cms.Sequence(
    patTauDiscriminationAgainstElectronMVA6Phase2Task
)
