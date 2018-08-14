import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Modifier_run2_miniAOD_80XLegacy_cff import run2_miniAOD_80XLegacy
from Configuration.Eras.Modifier_run2_nanoAOD_92X_cff import run2_nanoAOD_92X
from Configuration.Eras.Modifier_run2_nanoAOD_94XMiniAODv1_cff import run2_nanoAOD_94XMiniAODv1
from Configuration.Eras.Modifier_run2_nanoAOD_94XMiniAODv2_cff import run2_nanoAOD_94XMiniAODv2
from Configuration.Eras.Modifier_run2_nanoAOD_94X2016_cff import run2_nanoAOD_94X2016
from PhysicsTools.NanoAOD.common_cff import *
import PhysicsTools.PatAlgos.producersLayer1.electronProducer_cfi
from math import ceil,log

# this below is used only in some eras
slimmedElectronsUpdated = cms.EDProducer("PATElectronUpdater",
    src = cms.InputTag("slimmedElectrons"),
    vertices = cms.InputTag("offlineSlimmedPrimaryVertices"),
    computeMiniIso = cms.bool(False),
    pfCandsForMiniIso = cms.InputTag("packedPFCandidates"),
    miniIsoParamsB = PhysicsTools.PatAlgos.producersLayer1.electronProducer_cfi.patElectrons.miniIsoParamsB, # so they're in sync
    miniIsoParamsE = PhysicsTools.PatAlgos.producersLayer1.electronProducer_cfi.patElectrons.miniIsoParamsE, # so they're in sync
)
run2_miniAOD_80XLegacy.toModify( slimmedElectronsUpdated, computeMiniIso = True )

from PhysicsTools.SelectorUtils.tools.vid_id_tools import setupVIDSelection
from RecoEgamma.ElectronIdentification.egmGsfElectronIDs_cff import *
from RecoEgamma.ElectronIdentification.heepIdVarValueMapProducer_cfi import *

electronMVAValueMapProducer.srcMiniAOD = cms.InputTag("slimmedElectrons")
run2_miniAOD_80XLegacy.toModify(electronMVAValueMapProducer, srcMiniAOD = "slimmedElectronsUpdated")
run2_nanoAOD_92X.toModify(electronMVAValueMapProducer, srcMiniAOD = "slimmedElectronsUpdated")

electronMVAVariableHelper.srcMiniAOD = cms.InputTag("slimmedElectrons")
run2_miniAOD_80XLegacy.toModify(electronMVAVariableHelper, srcMiniAOD = "slimmedElectronsUpdated")
run2_nanoAOD_92X.toModify(electronMVAVariableHelper, srcMiniAOD = "slimmedElectronsUpdated")

egmGsfElectronIDs.physicsObjectIDs = cms.VPSet()
egmGsfElectronIDs.physicsObjectSrc = cms.InputTag('slimmedElectrons')
run2_miniAOD_80XLegacy.toModify(egmGsfElectronIDs, physicsObjectSrc = "slimmedElectronsUpdated")
run2_nanoAOD_92X.toModify(egmGsfElectronIDs, physicsObjectSrc = "slimmedElectronsUpdated")

heepIDVarValueMaps.elesMiniAOD = cms.InputTag('slimmedElectrons')
run2_miniAOD_80XLegacy.toModify(heepIDVarValueMaps, elesMiniAOD = "slimmedElectronsUpdated")
run2_nanoAOD_92X.toModify(heepIDVarValueMaps, elesMiniAOD = "slimmedElectronsUpdated")

_electron_id_modules_WorkingPoints = cms.PSet(
    modules = cms.vstring(
        'RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Fall17_94X_V1_cff',
        'RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Fall17_94X_V2_cff',
        'RecoEgamma.ElectronIdentification.Identification.heepElectronID_HEEPV70_cff',
        'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_iso_V1_cff',
        'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_noIso_V1_cff',
        'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_iso_V2_cff',
        'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_noIso_V2_cff',
    ),
    WorkingPoints = cms.vstring(
        "egmGsfElectronIDs:cutBasedElectronID-Fall17-94X-V2-veto",
        "egmGsfElectronIDs:cutBasedElectronID-Fall17-94X-V2-loose",
        "egmGsfElectronIDs:cutBasedElectronID-Fall17-94X-V2-medium",
        "egmGsfElectronIDs:cutBasedElectronID-Fall17-94X-V2-tight",
    )
)
run2_miniAOD_80XLegacy.toModify(_electron_id_modules_WorkingPoints,
    modules = cms.vstring(
        'RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Summer16_80X_V1_cff',
        'RecoEgamma.ElectronIdentification.Identification.cutBasedElectronHLTPreselecition_Summer16_V1_cff',
        'RecoEgamma.ElectronIdentification.Identification.heepElectronID_HEEPV70_cff',
        'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring16_GeneralPurpose_V1_cff',
        'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring16_HZZ_V1_cff',
    ),
    WorkingPoints = cms.vstring(
        "egmGsfElectronIDs:cutBasedElectronID-Summer16-80X-V1-loose",
        "egmGsfElectronIDs:cutBasedElectronID-Summer16-80X-V1-medium",
        "egmGsfElectronIDs:cutBasedElectronID-Summer16-80X-V1-tight",
    )
)
run2_nanoAOD_94X2016.toModify(_electron_id_modules_WorkingPoints,
    modules = cms.vstring(
        'RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Summer16_80X_V1_cff',
        'RecoEgamma.ElectronIdentification.Identification.cutBasedElectronHLTPreselecition_Summer16_V1_cff',
    ),
    WorkingPoints = cms.vstring(
        "egmGsfElectronIDs:cutBasedElectronID-Summer16-80X-V1-veto",
        "egmGsfElectronIDs:cutBasedElectronID-Summer16-80X-V1-loose",
        "egmGsfElectronIDs:cutBasedElectronID-Summer16-80X-V1-medium",
        "egmGsfElectronIDs:cutBasedElectronID-Summer16-80X-V1-tight",
    )
)
 

_bitmapVIDForEle_docstring = ''
for modname in _electron_id_modules_WorkingPoints.modules:
    ids= __import__(modname, globals(), locals(), ['idName','cutFlow'])
    for name in dir(ids):
        _id = getattr(ids,name)
        if hasattr(_id,'idName') and hasattr(_id,'cutFlow'):
            setupVIDSelection(egmGsfElectronIDs,_id)
            if (len(_electron_id_modules_WorkingPoints.WorkingPoints)>0 and _id.idName==_electron_id_modules_WorkingPoints.WorkingPoints[0].split(':')[-1]):
                _bitmapVIDForEle_docstring = 'VID compressed bitmap (%s), %d bits per cut'%(','.join([cut.cutName.value() for cut in _id.cutFlow]),int(ceil(log(len(_electron_id_modules_WorkingPoints.WorkingPoints)+1,2))))

bitmapVIDForEle = cms.EDProducer("EleVIDNestedWPBitmapProducer",
    src = cms.InputTag("slimmedElectrons"),
    WorkingPoints = _electron_id_modules_WorkingPoints.WorkingPoints,
)
run2_miniAOD_80XLegacy.toModify(bitmapVIDForEle, src = "slimmedElectronsUpdated")
run2_nanoAOD_92X.toModify(bitmapVIDForEle, src = "slimmedElectronsUpdated")

isoForEle = cms.EDProducer("EleIsoValueMapProducer",
    src = cms.InputTag("slimmedElectrons"),
    relative = cms.bool(False),
    rho_MiniIso = cms.InputTag("fixedGridRhoFastjetAll"),
    rho_PFIso = cms.InputTag("fixedGridRhoFastjetAll"),
    EAFile_MiniIso = cms.FileInPath("RecoEgamma/ElectronIdentification/data/Fall17/effAreaElectrons_cone03_pfNeuHadronsAndPhotons_94X.txt"),
    EAFile_PFIso = cms.FileInPath("RecoEgamma/ElectronIdentification/data/Fall17/effAreaElectrons_cone03_pfNeuHadronsAndPhotons_94X.txt"),
)
run2_miniAOD_80XLegacy.toModify(isoForEle, src = "slimmedElectronsUpdated",
                                EAFile_MiniIso = "RecoEgamma/ElectronIdentification/data/Spring15/effAreaElectrons_cone03_pfNeuHadronsAndPhotons_25ns.txt",
                                EAFile_PFIso = "RecoEgamma/ElectronIdentification/data/Summer16/effAreaElectrons_cone03_pfNeuHadronsAndPhotons_80X.txt")
run2_nanoAOD_92X.toModify(isoForEle, src = "slimmedElectronsUpdated")

ptRatioRelForEle = cms.EDProducer("ElectronJetVarProducer",
    srcJet = cms.InputTag("updatedJets"),
    srcLep = cms.InputTag("slimmedElectrons"),
    srcVtx = cms.InputTag("offlineSlimmedPrimaryVertices"),
)
run2_miniAOD_80XLegacy.toModify(ptRatioRelForEle, srcLep = "slimmedElectronsUpdated")
run2_nanoAOD_92X.toModify(ptRatioRelForEle, srcLep = "slimmedElectronsUpdated")

import EgammaAnalysis.ElectronTools.calibratedElectronsRun2_cfi
calibratedPatElectrons80X = EgammaAnalysis.ElectronTools.calibratedElectronsRun2_cfi.calibratedPatElectrons.clone(
    electrons = cms.InputTag("slimmedElectronsUpdated"),
    correctionFile = cms.string("PhysicsTools/NanoAOD/data/80X_ichepV1_2016_ele"),
    semiDeterministic = cms.bool(True)
)
energyCorrForEle80X =  cms.EDProducer("ElectronEnergyVarProducer",
    srcRaw = cms.InputTag("slimmedElectronsUpdated"),
    srcCorr = cms.InputTag("calibratedPatElectrons80X"),
)
import RecoEgamma.EgammaTools.calibratedEgammas_cff
calibratedPatElectrons94Xv1 = RecoEgamma.EgammaTools.calibratedEgammas_cff.calibratedPatElectrons.clone(
    produceCalibratedObjs = False
)

slimmedElectronsWithUserData = cms.EDProducer("PATElectronUserDataEmbedder",
    src = cms.InputTag("slimmedElectrons"),
    userFloats = cms.PSet(
        mvaFall17V1Iso = cms.InputTag("electronMVAValueMapProducer:ElectronMVAEstimatorRun2Fall17IsoV1Values"),
        mvaFall17V1noIso = cms.InputTag("electronMVAValueMapProducer:ElectronMVAEstimatorRun2Fall17NoIsoV1Values"),
        mvaFall17V2Iso = cms.InputTag("electronMVAValueMapProducer:ElectronMVAEstimatorRun2Fall17IsoV2Values"),
        mvaFall17V2noIso = cms.InputTag("electronMVAValueMapProducer:ElectronMVAEstimatorRun2Fall17NoIsoV2Values"),
        miniIsoChg = cms.InputTag("isoForEle:miniIsoChg"),
        miniIsoAll = cms.InputTag("isoForEle:miniIsoAll"),
        PFIsoChg = cms.InputTag("isoForEle:PFIsoChg"),
        PFIsoAll = cms.InputTag("isoForEle:PFIsoAll"),
        PFIsoAll04 = cms.InputTag("isoForEle:PFIsoAll04"),
        ptRatio = cms.InputTag("ptRatioRelForEle:ptRatio"),
        ptRel = cms.InputTag("ptRatioRelForEle:ptRel"),
        jetNDauChargedMVASel = cms.InputTag("ptRatioRelForEle:jetNDauChargedMVASel"),
    ),
    userIntFromBools = cms.PSet(

        mvaFall17V1Iso_WP90 = cms.InputTag("egmGsfElectronIDs:mvaEleID-Fall17-iso-V1-wp90"),
        mvaFall17V1Iso_WP80 = cms.InputTag("egmGsfElectronIDs:mvaEleID-Fall17-iso-V1-wp80"),
        mvaFall17V1Iso_WPL = cms.InputTag("egmGsfElectronIDs:mvaEleID-Fall17-iso-V1-wpLoose"),
        mvaFall17V1noIso_WP90 = cms.InputTag("egmGsfElectronIDs:mvaEleID-Fall17-noIso-V1-wp90"),
        mvaFall17V1noIso_WP80 = cms.InputTag("egmGsfElectronIDs:mvaEleID-Fall17-noIso-V1-wp80"),
        mvaFall17V1noIso_WPL = cms.InputTag("egmGsfElectronIDs:mvaEleID-Fall17-noIso-V1-wpLoose"),

        mvaFall17V2Iso_WP90 = cms.InputTag("egmGsfElectronIDs:mvaEleID-Fall17-iso-V2-wp90"),
        mvaFall17V2Iso_WP80 = cms.InputTag("egmGsfElectronIDs:mvaEleID-Fall17-iso-V2-wp80"),
        mvaFall17V2Iso_WPL = cms.InputTag("egmGsfElectronIDs:mvaEleID-Fall17-iso-V2-wpLoose"),
        mvaFall17V2noIso_WP90 = cms.InputTag("egmGsfElectronIDs:mvaEleID-Fall17-noIso-V2-wp90"),
        mvaFall17V2noIso_WP80 = cms.InputTag("egmGsfElectronIDs:mvaEleID-Fall17-noIso-V2-wp80"),
        mvaFall17V2noIso_WPL = cms.InputTag("egmGsfElectronIDs:mvaEleID-Fall17-noIso-V2-wpLoose"),

        cutbasedID_Fall17_V1_veto = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Fall17-94X-V1-veto"),
        cutbasedID_Fall17_V1_loose = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Fall17-94X-V1-loose"),
        cutbasedID_Fall17_V1_medium = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Fall17-94X-V1-medium"),
        cutbasedID_Fall17_V1_tight = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Fall17-94X-V1-tight"),
        cutbasedID_veto = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Fall17-94X-V2-veto"),
        cutbasedID_loose = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Fall17-94X-V2-loose"),
        cutbasedID_medium = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Fall17-94X-V2-medium"),
        cutbasedID_tight = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Fall17-94X-V2-tight"),
        cutbasedID_HEEP = cms.InputTag("egmGsfElectronIDs:heepElectronID-HEEPV70"),
    ),
    userInts = cms.PSet(
        VIDNestedWPBitmap = cms.InputTag("bitmapVIDForEle"),
    ),
    userCands = cms.PSet(
        jetForLepJetVar = cms.InputTag("ptRatioRelForEle:jetForLepJetVar") # warning: Ptr is null if no match is found
    ),
)
run2_miniAOD_80XLegacy.toModify(slimmedElectronsWithUserData, src = "slimmedElectronsUpdated")
run2_nanoAOD_92X.toModify(slimmedElectronsWithUserData, src = "slimmedElectronsUpdated")
run2_miniAOD_80XLegacy.toModify(slimmedElectronsWithUserData.userFloats,
    mvaSpring16GP = cms.InputTag("electronMVAValueMapProducer:ElectronMVAEstimatorRun2Spring16GeneralPurposeV1Values"),
    mvaSpring16HZZ = cms.InputTag("electronMVAValueMapProducer:ElectronMVAEstimatorRun2Spring16HZZV1Values"),
    mvaFall17V1Iso = None,
    mvaFall17V1noIso = None,
    mvaFall17V2Iso = None,
    mvaFall17V2noIso = None,
    eCorr = cms.InputTag("energyCorrForEle80X","eCorr")
)
run2_nanoAOD_94X2016.toModify(slimmedElectronsWithUserData.userFloats,
    mvaFall17V1Iso = None,
    mvaFall17V1noIso = None,
    mvaFall17V2Iso = None,
    mvaFall17V2noIso = None,
)
run2_nanoAOD_94XMiniAODv1.toModify(slimmedElectronsWithUserData.userFloats,
    ecalTrkEnergyErrPostCorr = cms.InputTag("calibratedPatElectrons94Xv1","ecalTrkEnergyErrPostCorr"),
    ecalTrkEnergyPreCorr     = cms.InputTag("calibratedPatElectrons94Xv1","ecalTrkEnergyPreCorr"),
    ecalTrkEnergyPostCorr    = cms.InputTag("calibratedPatElectrons94Xv1","ecalTrkEnergyPostCorr"),
)
run2_miniAOD_80XLegacy.toReplaceWith(slimmedElectronsWithUserData.userIntFromBools,
    cms.PSet(
        mvaSpring16GP_WP90 = cms.InputTag("egmGsfElectronIDs:mvaEleID-Spring16-GeneralPurpose-V1-wp90"),
        mvaSpring16GP_WP80 = cms.InputTag("egmGsfElectronIDs:mvaEleID-Spring16-GeneralPurpose-V1-wp80"),
        mvaSpring16HZZ_WPL = cms.InputTag("egmGsfElectronIDs:mvaEleID-Spring16-HZZ-V1-wpLoose"),
        cutbasedID_veto = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Summer16-80X-V1-veto"),
        cutbasedID_loose = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Summer16-80X-V1-loose"),
        cutbasedID_medium = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Summer16-80X-V1-medium"),
        cutbasedID_tight = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Summer16-80X-V1-tight"),
        cutbasedID_HLT = cms.InputTag("egmGsfElectronIDs:cutBasedElectronHLTPreselection-Summer16-V1"),
        cutbasedID_HEEP = cms.InputTag("egmGsfElectronIDs:heepElectronID-HEEPV70"),
    )
)
run2_nanoAOD_94X2016.toReplaceWith(slimmedElectronsWithUserData.userIntFromBools,
    cms.PSet(
        # MVAs and HEEP are already pre-computed. Cut-based too, but we re-add it for consistency with the nested bitmap
        cutbasedID_veto = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Summer16-80X-V1-veto"),
        cutbasedID_loose = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Summer16-80X-V1-loose"),
        cutbasedID_medium = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Summer16-80X-V1-medium"),
        cutbasedID_tight = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Summer16-80X-V1-tight"),
        cutbasedID_HLT = cms.InputTag("egmGsfElectronIDs:cutBasedElectronHLTPreselection-Summer16-V1"),
    )
)
finalElectrons = cms.EDFilter("PATElectronRefSelector",
    src = cms.InputTag("slimmedElectronsWithUserData"),
    cut = cms.string("pt > 5 ")
)

electronMVATTH= cms.EDProducer("EleBaseMVAValueMapProducer",
    src = cms.InputTag("linkedObjects","electrons"),
    weightFile =  cms.FileInPath("PhysicsTools/NanoAOD/data/el_BDTG_94X.weights.xml"),
    name = cms.string("electronMVATTH"),
    isClassifier = cms.bool(True),
    variablesOrder = cms.vstring(["LepGood_pt","LepGood_eta","LepGood_jetNDauChargedMVASel","LepGood_miniRelIsoCharged","LepGood_miniRelIsoNeutral","LepGood_jetPtRelv2","LepGood_jetBTagCSV","LepGood_jetPtRatio","LepGood_sip3d","LepGood_dxy","LepGood_dz","LepGood_mvaIdFall17noIso"]),
    variables = cms.PSet(
        LepGood_pt = cms.string("pt"),
        LepGood_eta = cms.string("eta"),
        LepGood_jetNDauChargedMVASel = cms.string("?userCand('jetForLepJetVar').isNonnull()?userFloat('jetNDauChargedMVASel'):0"),
        LepGood_miniRelIsoCharged = cms.string("userFloat('miniIsoChg')/pt"),
        LepGood_miniRelIsoNeutral = cms.string("(userFloat('miniIsoAll')-userFloat('miniIsoChg'))/pt"),
        LepGood_jetPtRelv2 = cms.string("?userCand('jetForLepJetVar').isNonnull()?userFloat('ptRel'):0"),
        LepGood_jetPtRatio = cms.string("?userCand('jetForLepJetVar').isNonnull()?min(userFloat('ptRatio'),1.5):1.0/(1.0+userFloat('PFIsoAll04')/pt)"),
        LepGood_jetBTagCSV = cms.string("?userCand('jetForLepJetVar').isNonnull()?max(userCand('jetForLepJetVar').bDiscriminator('pfCombinedInclusiveSecondaryVertexV2BJetTags'),0.0):0.0"),
        LepGood_sip3d = cms.string("abs(dB('PV3D')/edB('PV3D'))"),
        LepGood_dxy = cms.string("log(abs(dB('PV2D')))"),
        LepGood_dz = cms.string("log(abs(dB('PVDZ')))"),
        LepGood_mvaIdFall17noIso = cms.string("userFloat('mvaFall17V1noIso')"),
    )
)
for modifier in run2_miniAOD_80XLegacy, run2_nanoAOD_94X2016:
  modifier.toModify(electronMVATTH.variables,
    LepGood_jetPtRatio = cms.string("?userCand('jetForLepJetVar').isNonnull()?min(userFloat('ptRatio'),1.5):1"),
    LepGood_mvaIdSpring16HZZ = cms.string("userFloat('%s')" % ('mvaSpring16HZZ' if modifier == run2_miniAOD_80XLegacy else 'ElectronMVAEstimatorRun2Spring16HZZV1Values')),
    LepGood_mvaIdFall17noIso = None)
  modifier.toModify(electronMVATTH,
    weightFile = "PhysicsTools/NanoAOD/data/el_BDTG.weights.xml",
    variablesOrder = ["LepGood_pt","LepGood_eta","LepGood_jetNDauChargedMVASel","LepGood_miniRelIsoCharged","LepGood_miniRelIsoNeutral","LepGood_jetPtRelv2","LepGood_jetPtRatio","LepGood_jetBTagCSV","LepGood_sip3d","LepGood_dxy","LepGood_dz","LepGood_mvaIdSpring16HZZ"])

electronTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
    src = cms.InputTag("linkedObjects","electrons"),
    cut = cms.string(""), #we should not filter on cross linked collections
    name= cms.string("Electron"),
    doc = cms.string("slimmedElectrons after basic selection (" + finalElectrons.cut.value()+")"),
    singleton = cms.bool(False), # the number of entries is variable
    extension = cms.bool(False), # this is the main table for the electrons
    variables = cms.PSet(CandVars,
        jetIdx = Var("?hasUserCand('jet')?userCand('jet').key():-1", int, doc="index of the associated jet (-1 if none)"),
        photonIdx = Var("?overlaps('photons').size()>0?overlaps('photons')[0].key():-1", int, doc="index of the associated photon (-1 if none)"),
        energyErr = Var("p4Error('P4_COMBINATION')",float,doc="energy error of the cluster-track combination",precision=6),
        dz = Var("dB('PVDZ')",float,doc="dz (with sign) wrt first PV, in cm",precision=10),
        dzErr = Var("abs(edB('PVDZ'))",float,doc="dz uncertainty, in cm",precision=6),
        dxy = Var("dB('PV2D')",float,doc="dxy (with sign) wrt first PV, in cm",precision=10),
        dxyErr = Var("edB('PV2D')",float,doc="dxy uncertainty, in cm",precision=6),
        ip3d = Var("abs(dB('PV3D'))",float,doc="3D impact parameter wrt first PV, in cm",precision=10),
        sip3d = Var("abs(dB('PV3D')/edB('PV3D'))",float,doc="3D impact parameter significance wrt first PV, in cm",precision=10),
        deltaEtaSC = Var("superCluster().eta()-eta()",float,doc="delta eta (SC,ele) with sign",precision=10),
        r9 = Var("full5x5_r9()",float,doc="R9 of the supercluster, calculated with full 5x5 region",precision=10),
        sieie = Var("full5x5_sigmaIetaIeta()",float,doc="sigma_IetaIeta of the supercluster, calculated with full 5x5 region",precision=10),
        eInvMinusPInv = Var("(1-eSuperClusterOverP())/ecalEnergy()",float,doc="1/E_SC - 1/p_trk",precision=10),

        mvaFall17V1Iso = Var("userFloat('mvaFall17V1Iso')",float,doc="MVA Iso ID V1 score"),
        mvaFall17V1Iso_WP80 = Var("userInt('mvaFall17V1Iso_WP80')",bool,doc="MVA Iso ID V1 WP80"),
        mvaFall17V1Iso_WP90 = Var("userInt('mvaFall17V1Iso_WP90')",bool,doc="MVA Iso ID V1 WP90"),
        mvaFall17V1Iso_WPL = Var("userInt('mvaFall17V1Iso_WPL')",bool,doc="MVA Iso ID V1 loose WP"),
        mvaFall17V1noIso = Var("userFloat('mvaFall17V1noIso')",float,doc="MVA noIso ID V1 score"),
        mvaFall17V1noIso_WP80 = Var("userInt('mvaFall17V1noIso_WP80')",bool,doc="MVA noIso ID V1 WP80"),
        mvaFall17V1noIso_WP90 = Var("userInt('mvaFall17V1noIso_WP90')",bool,doc="MVA noIso ID V1 WP90"),
        mvaFall17V1noIso_WPL = Var("userInt('mvaFall17V1noIso_WPL')",bool,doc="MVA noIso ID V1 loose WP"),

        mvaFall17V2Iso = Var("userFloat('mvaFall17V2Iso')",float,doc="MVA Iso ID V2 score"),
        mvaFall17V2Iso_WP80 = Var("userInt('mvaFall17V2Iso_WP80')",bool,doc="MVA Iso ID V2 WP80"),
        mvaFall17V2Iso_WP90 = Var("userInt('mvaFall17V2Iso_WP90')",bool,doc="MVA Iso ID V2 WP90"),
        mvaFall17V2Iso_WPL = Var("userInt('mvaFall17V2Iso_WPL')",bool,doc="MVA Iso ID V2 loose WP"),
        mvaFall17V2noIso = Var("userFloat('mvaFall17V2noIso')",float,doc="MVA noIso ID V2 score"),
        mvaFall17V2noIso_WP80 = Var("userInt('mvaFall17V2noIso_WP80')",bool,doc="MVA noIso ID V2 WP80"),
        mvaFall17V2noIso_WP90 = Var("userInt('mvaFall17V2noIso_WP90')",bool,doc="MVA noIso ID V2 WP90"),
        mvaFall17V2noIso_WPL = Var("userInt('mvaFall17V2noIso_WPL')",bool,doc="MVA noIso ID V2 loose WP"),

        cutBased = Var("userInt('cutbasedID_veto')+userInt('cutbasedID_loose')+userInt('cutbasedID_medium')+userInt('cutbasedID_tight')",int,doc="cut-based ID (0:fail, 1:veto, 2:loose, 3:medium, 4:tight)"),
        cutBased_Fall17_V1 = Var("userInt('cutbasedID_Fall17_V1_veto')+userInt('cutbasedID_Fall17_V1_loose')+userInt('cutbasedID_Fall17_V1_medium')+userInt('cutbasedID_Fall17_V1_tight')",int,doc="cut-based ID Fall17 V1 (0:fail, 1:veto, 2:loose, 3:medium, 4:tight)"),
        vidNestedWPBitmap = Var("userInt('VIDNestedWPBitmap')",int,doc=_bitmapVIDForEle_docstring),
        cutBased_HEEP = Var("userInt('cutbasedID_HEEP')",bool,doc="cut-based HEEP ID"),
        miniPFRelIso_chg = Var("userFloat('miniIsoChg')/pt",float,doc="mini PF relative isolation, charged component"),
        miniPFRelIso_all = Var("userFloat('miniIsoAll')/pt",float,doc="mini PF relative isolation, total (with scaled rho*EA PU corrections)"),
        pfRelIso03_chg = Var("userFloat('PFIsoChg')/pt",float,doc="PF relative isolation dR=0.3, charged component"),
        pfRelIso03_all = Var("userFloat('PFIsoAll')/pt",float,doc="PF relative isolation dR=0.3, total (with rho*EA PU corrections)"),
        dr03TkSumPt = Var("?pt>35?dr03TkSumPt():0",float,doc="Non-PF track isolation within a delta R cone of 0.3 with electron pt > 35 GeV",precision=8),
        dr03EcalRecHitSumEt = Var("?pt>35?dr03EcalRecHitSumEt():0",float,doc="Non-PF Ecal isolation within a delta R cone of 0.3 with electron pt > 35 GeV",precision=8),
        dr03HcalDepth1TowerSumEt = Var("?pt>35?dr03HcalDepth1TowerSumEt():0",float,doc="Non-PF Hcal isolation within a delta R cone of 0.3 with electron pt > 35 GeV",precision=8),
        hoe = Var("hadronicOverEm()",float,doc="H over E",precision=8),
        tightCharge = Var("isGsfCtfScPixChargeConsistent() + isGsfScPixChargeConsistent()",int,doc="Tight charge criteria (0:none, 1:isGsfScPixChargeConsistent, 2:isGsfCtfScPixChargeConsistent)"),
        convVeto = Var("passConversionVeto()",bool,doc="pass conversion veto"),
        lostHits = Var("gsfTrack.hitPattern.numberOfLostHits('MISSING_INNER_HITS')","uint8",doc="number of missing inner hits"),
        isPFcand = Var("pfCandidateRef().isNonnull()",bool,doc="electron is PF candidate"),
    ),
    externalVariables = cms.PSet(
        mvaTTH = ExtVar(cms.InputTag("electronMVATTH"),float, doc="TTH MVA lepton ID score",precision=14),
    ),
)
# scale and smearing only when available
for modifier in run2_nanoAOD_94XMiniAODv1, run2_nanoAOD_94XMiniAODv2, run2_nanoAOD_94X2016:
    modifier.toModify(electronTable.variables,
        pt = Var("pt*userFloat('ecalTrkEnergyPostCorr')/userFloat('ecalTrkEnergyPreCorr')", float, precision=-1, doc="p_{T}"),
        energyErr = Var("userFloat('ecalTrkEnergyErrPostCorr')", float, precision=6, doc="energy error of the cluster-track combination"),
        eCorr = Var("userFloat('ecalTrkEnergyPostCorr')/userFloat('ecalTrkEnergyPreCorr')", float, doc="ratio of the calibrated energy/miniaod energy"),
    )
run2_nanoAOD_94X2016.toModify(electronTable.variables,
    cutBased = Var("userInt('cutbasedID_veto')+userInt('cutbasedID_loose')+userInt('cutbasedID_medium')+userInt('cutbasedID_tight')",int,doc="cut-based Summer16 ID (0:fail, 1:veto, 2:loose, 3:medium, 4:tight)"),
    cutBased_Fall17_V1 = Var("electronID('cutBasedElectronID-Fall17-94X-V1-veto')+electronID('cutBasedElectronID-Fall17-94X-V1-loose')+electronID('cutBasedElectronID-Fall17-94X-V1-medium')+electronID('cutBasedElectronID-Fall17-94X-V1-tight')",int,doc="cut-based Fall17 ID (0:fail, 1:veto, 2:loose, 3:medium, 4:tight)"),
    cutBased_HLTPreSel = Var("userInt('cutbasedID_HLT')",int,doc="cut-based HLT pre-selection ID"),
    cutBased_HEEP = Var("electronID('heepElectronID-HEEPV70')",bool,doc="cut-based HEEP ID"),
    mvaSpring16GP = Var("userFloat('ElectronMVAEstimatorRun2Spring16GeneralPurposeV1Values')",float,doc="MVA Spring16 general-purpose ID score"),
    mvaSpring16GP_WP80 = Var("electronID('mvaEleID-Spring16-GeneralPurpose-V1-wp80')",bool,doc="MVA Spring16 general-purpose ID WP80"),
    mvaSpring16GP_WP90 = Var("electronID('mvaEleID-Spring16-GeneralPurpose-V1-wp90')",bool,doc="MVA Spring16 general-purpose ID WP90"),
    mvaSpring16HZZ = Var("userFloat('ElectronMVAEstimatorRun2Spring16HZZV1Values')",float,doc="MVA Spring16 HZZ ID score"),
    mvaSpring16HZZ_WPL = Var("electronID('mvaEleID-Spring16-HZZ-V1-wpLoose')",bool,doc="MVA Spring16 HZZ ID loose WP"),
    mvaFall17V1Iso = Var("userFloat('ElectronMVAEstimatorRun2Fall17IsoV1Values')",float,doc="MVA Fall17 V1 Iso ID score"),
    mvaFall17V1Iso_WP80 = Var("electronID('mvaEleID-Fall17-iso-V1-wp80')",bool,doc="MVA Fall17 V1 Iso ID WP80"),
    mvaFall17V1Iso_WP90 = Var("electronID('mvaEleID-Fall17-iso-V1-wp90')",bool,doc="MVA Fall17 V1 Iso ID WP90"),
    mvaFall17V1Iso_WPL = Var("electronID('mvaEleID-Fall17-iso-V1-wpLoose')",bool,doc="MVA Fall17 V1 Iso ID loose WP"),
    mvaFall17V1noIso = Var("userFloat('ElectronMVAEstimatorRun2Fall17NoIsoV1Values')",float,doc="MVA Fall17 V1 noIso ID score"),
    mvaFall17V1noIso_WP80 = Var("electronID('mvaEleID-Fall17-noIso-V1-wp80')",bool,doc="MVA Fall17 V1 noIso ID WP80"),
    mvaFall17V1noIso_WP90 = Var("electronID('mvaEleID-Fall17-noIso-V1-wp90')",bool,doc="MVA Fall17 V1 noIso ID WP90"),
    mvaFall17V1noIso_WPL = Var("electronID('mvaEleID-Fall17-noIso-V1-wpLoose')",bool,doc="MVA Fall17 V1 noIso ID loose WP"),
    mvaFall17V2Iso = None,
    mvaFall17V2Iso_WP80 = None,
    mvaFall17V2Iso_WP90 = None,
    mvaFall17V2Iso_WPL = None,
    mvaFall17V2noIso = None,
    mvaFall17V2noIso_WP80 = None,
    mvaFall17V2noIso_WP90 = None,
    mvaFall17V2noIso_WPL = None,
)
run2_miniAOD_80XLegacy.toModify(electronTable.variables,
    cutBased_HLTPreSel = Var("userInt('cutbasedID_HLT')",int,doc="cut-based HLT pre-selection ID"),
    mvaSpring16GP = Var("userFloat('mvaSpring16GP')",float,doc="MVA general-purpose ID score"),
    mvaSpring16GP_WP80 = Var("userInt('mvaSpring16GP_WP80')",bool,doc="MVA general-purpose ID WP80"),
    mvaSpring16GP_WP90 = Var("userInt('mvaSpring16GP_WP90')",bool,doc="MVA general-purpose ID WP90"),
    mvaSpring16HZZ = Var("userFloat('mvaSpring16HZZ')",float,doc="MVA HZZ ID score"),
    mvaSpring16HZZ_WPL = Var("userInt('mvaSpring16HZZ_WPL')",bool,doc="MVA HZZ ID loose WP"),

    mvaFall17V1Iso = None,
    mvaFall17V1Iso_WP80 = None,
    mvaFall17V1Iso_WP90 = None,
    mvaFall17V1Iso_WPL = None,
    mvaFall17V1noIso = None,
    mvaFall17V1noIso_WP80 = None,
    mvaFall17V1noIso_WP90 = None,
    mvaFall17V1noIso_WPL = None,

    mvaFall17V2Iso = None,
    mvaFall17V2Iso_WP80 = None,
    mvaFall17V2Iso_WP90 = None,
    mvaFall17V2Iso_WPL = None,
    mvaFall17V2noIso = None,
    mvaFall17V2noIso_WP80 = None,
    mvaFall17V2noIso_WP90 = None,
    mvaFall17V2noIso_WPL = None,

    cutBased_Fall17_V1 = None,

    pt = Var("pt*userFloat('eCorr')",  float, precision=-1, doc="p_{T} after energy correction & smearing"),
    energyErr = Var("p4Error('P4_COMBINATION')*userFloat('eCorr')",float,doc="energy error of the cluster-track combination",precision=6),
    eCorr = Var("userFloat('eCorr')",float,doc="ratio of the calibrated energy/miniaod energy"),
)

electronsMCMatchForTable = cms.EDProducer("MCMatcher",  # cut on deltaR, deltaPt/Pt; pick best by deltaR
    src         = electronTable.src,                 # final reco collection
    matched     = cms.InputTag("finalGenParticles"), # final mc-truth particle collection
    mcPdgId     = cms.vint32(11,22),                 # one or more PDG ID (11 = el, 22 = pho); absolute values (see below)
    checkCharge = cms.bool(False),              # True = require RECO and MC objects to have the same charge
    mcStatus    = cms.vint32(1),                # PYTHIA status code (1 = stable, 2 = shower, 3 = hard scattering)
    maxDeltaR   = cms.double(0.3),              # Minimum deltaR for the match
    maxDPtRel   = cms.double(0.5),              # Minimum deltaPt/Pt for the match
    resolveAmbiguities    = cms.bool(True),     # Forbid two RECO objects to match to the same GEN object
    resolveByMatchQuality = cms.bool(True),    # False = just match input in order; True = pick lowest deltaR pair first
)

electronMCTable = cms.EDProducer("CandMCMatchTableProducer",
    src     = electronTable.src,
    mcMap   = cms.InputTag("electronsMCMatchForTable"),
    objName = electronTable.name,
    objType = electronTable.name, #cms.string("Electron"),
    branchName = cms.string("genPart"),
    docString = cms.string("MC matching to status==1 electrons or photons"),
)

electronSequence = cms.Sequence(heepIDVarValueMaps + egmGsfElectronIDSequence + bitmapVIDForEle + isoForEle + ptRatioRelForEle + slimmedElectronsWithUserData + finalElectrons)
electronTables = cms.Sequence (electronMVATTH + electronTable)
electronMC = cms.Sequence(electronsMCMatchForTable + electronMCTable)

_withUpdate_sequence = cms.Sequence(slimmedElectronsUpdated + electronSequence.copy())
run2_nanoAOD_92X.toReplaceWith(electronSequence, _withUpdate_sequence)

_withUpdateAnd80XScale_sequence = _withUpdate_sequence.copy()
_withUpdateAnd80XScale_sequence.replace(slimmedElectronsWithUserData, calibratedPatElectrons80X + energyCorrForEle80X + slimmedElectronsWithUserData)
run2_miniAOD_80XLegacy.toReplaceWith(electronSequence, _withUpdateAnd80XScale_sequence)

_with94Xv1Scale_sequence = electronSequence.copy()
_with94Xv1Scale_sequence.replace(slimmedElectronsWithUserData, calibratedPatElectrons94Xv1 + slimmedElectronsWithUserData)
run2_nanoAOD_94XMiniAODv1.toReplaceWith(electronSequence, _with94Xv1Scale_sequence)

