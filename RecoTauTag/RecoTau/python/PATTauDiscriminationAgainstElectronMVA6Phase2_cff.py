import FWCore.ParameterSet.Config as cms

# Electron collection merger
mergedSlimmedElectronsForTauId = cms.EDProducer('PATElectronCollectionMerger',
    src = cms.VInputTag('slimmedElectrons', 'slimmedElectronsFromMultiCl')
)
# anti-e phase-2 tauID (Raw)
from RecoTauTag.RecoTau.TauDiscriminatorTools import noPrediscriminants
from RecoTauTag.RecoTau.patTauDiscriminationAgainstElectronMVA6_cfi import *
patTauDiscriminationAgainstElectronMVA6Phase2Raw = patTauDiscriminationAgainstElectronMVA6.clone(
    #Note: PATTauProducer has to be set in the final cfg
    Prediscriminants = noPrediscriminants,
    srcElectrons = "mergedSlimmedElectronsForTauId",
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
# anti-e phase-2 tauID (WPs)
from RecoTauTag.RecoTau.patTauDiscriminantCutMultiplexerDefault_cfi import patTauDiscriminantCutMultiplexerDefault
patTauDiscriminationAgainstElectronMVA6Phase2 = patTauDiscriminantCutMultiplexerDefault.clone(
    #Note: PFTauProducer and Prediscriminants have to be set in the final cfg
    toMultiplex = 'patTauDiscriminationAgainstElectronMVA6Phase2Raw',
    mapping = [
        cms.PSet(
            category = cms.uint32(0), # minMVANoEleMatchWOgWOgsfBL
            cut = cms.string("RecoTauTag_antiElectronPhase2MVA6v1_gbr_NoEleMatch_woGwoGSF_BL"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(2), # minMVANoEleMatchWgWOgsfBL
            cut = cms.string("RecoTauTag_antiElectronPhase2MVA6v1_gbr_NoEleMatch_wGwoGSF_BL"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(5), # minMVAWOgWgsfBL
            cut = cms.string("RecoTauTag_antiElectronPhase2MVA6v1_gbr_woGwGSF_BL"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(7), # minMVAWgWgsfBL
            cut = cms.string("RecoTauTag_antiElectronPhase2MVA6v1_gbr_wGwGSF_BL"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(8), # minMVANoEleMatchWOgWOgsfEC
            cut = cms.string("RecoTauTag_antiElectronPhase2MVA6v1_gbr_NoEleMatch_woGwoGSF_FWEC"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(9), # minMVANoEleMatchWOgWOgsfVFEC
            cut = cms.string("RecoTauTag_antiElectronPhase2MVA6v1_gbr_NoEleMatch_woGwoGSF_VFWEC"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(10), # minMVANoEleMatchWgWOgsfEC
            cut = cms.string("RecoTauTag_antiElectronPhase2MVA6v1_gbr_NoEleMatch_wGwoGSF_FWEC"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(11), # minMVANoEleMatchWgWOgsfVFEC
            cut = cms.string("RecoTauTag_antiElectronPhase2MVA6v1_gbr_NoEleMatch_wGwoGSF_VFWEC"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(13), # minMVAWOgWgsfEC
            cut = cms.string("RecoTauTag_antiElectronPhase2MVA6v1_gbr_woGwGSF_FWEC"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(14), # minMVAWOgWgsfVFEC
            cut = cms.string("RecoTauTag_antiElectronPhase2MVA6v1_gbr_woGwGSF_VFWEC"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(15), # minMVAWgWgsfEC
            cut = cms.string("RecoTauTag_antiElectronPhase2MVA6v1_gbr_wGwGSF_FWEC"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(16), # minMVAWgWgsfVFEC
            cut = cms.string("RecoTauTag_antiElectronPhase2MVA6v1_gbr_wGwGSF_VFWEC"),
            variable = cms.string("pt")
        )
    ],
    rawValues = ["discriminator", "category"],
    workingPoints = cms.vstring(
        "_WPEff98",
        "_WPEff90",
        "_WPEff80",
        "_WPEff70",
        "_WPEff60"
    )
)

patTauDiscriminationAgainstElectronMVA6Phase2Task = cms.Task(
    mergedSlimmedElectronsForTauId,
    patTauDiscriminationAgainstElectronMVA6Phase2Raw,
    patTauDiscriminationAgainstElectronMVA6Phase2
)

patTauDiscriminationAgainstElectronMVA6Phase2Seq = cms.Sequence(
    patTauDiscriminationAgainstElectronMVA6Phase2Task
)
