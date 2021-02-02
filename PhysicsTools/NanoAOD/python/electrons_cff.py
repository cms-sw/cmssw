import FWCore.ParameterSet.Config as cms

from PhysicsTools.NanoAOD.nano_eras_cff import *
from PhysicsTools.NanoAOD.common_cff import *
import PhysicsTools.PatAlgos.producersLayer1.electronProducer_cfi
from math import ceil,log

from RecoEgamma.EgammaTools.egammaObjectModificationsInMiniAOD_cff import egamma8XObjectUpdateModifier,egamma9X105XUpdateModifier,prependEgamma8XObjectUpdateModifier
ele9X105XUpdateModifier=egamma9X105XUpdateModifier.clone(
    phoPhotonIso = "",
    phoNeutralHadIso = "",
    phoChargedHadIso = "",
    phoChargedHadWorstVtxIso = "",
    phoChargedHadWorstVtxConeVetoIso = "",
    phoChargedHadPFPVIso = ""
)
#we have dataformat changes to 106X so to read older releases we use egamma updators
slimmedElectronsTo106X = cms.EDProducer("ModifiedElectronProducer",
    src = cms.InputTag("slimmedElectrons"),
    modifierConfig = cms.PSet( modifications = cms.VPSet(ele9X105XUpdateModifier) )
)
#might as well fix 80X while we're at it although the differences are not so relavent for nano
run2_miniAOD_80XLegacy.toModify( slimmedElectronsTo106X.modifierConfig.modifications, prependEgamma8XObjectUpdateModifier )

# this below is used only in some eras
slimmedElectronsUpdated = cms.EDProducer("PATElectronUpdater",
    src = cms.InputTag("slimmedElectronsTo106X"),
    vertices = cms.InputTag("offlineSlimmedPrimaryVertices"),
    computeMiniIso = cms.bool(False),
    fixDxySign = cms.bool(True),
    pfCandsForMiniIso = cms.InputTag("packedPFCandidates"),
    miniIsoParamsB = PhysicsTools.PatAlgos.producersLayer1.electronProducer_cfi.patElectrons.miniIsoParamsB, # so they're in sync
    miniIsoParamsE = PhysicsTools.PatAlgos.producersLayer1.electronProducer_cfi.patElectrons.miniIsoParamsE, # so they're in sync
)
run2_miniAOD_80XLegacy.toModify( slimmedElectronsUpdated, computeMiniIso = True )
# bypass the update to 106X in 106X to only pick up the IP sign fix
run2_egamma_2017.toModify(slimmedElectronsUpdated, src = cms.InputTag("slimmedElectrons"))
run2_egamma_2018.toModify(slimmedElectronsUpdated, src = cms.InputTag("slimmedElectrons"))
run2_nanoAOD_106Xv1.toModify(slimmedElectronsUpdated, src = cms.InputTag("slimmedElectrons"))
####because run2_egamma_2017 and run2_egamma_2018 can modify things further, need the following line to resort back
for modifier in run2_miniAOD_80XLegacy,run2_nanoAOD_94X2016,run2_nanoAOD_94XMiniAODv1,run2_nanoAOD_94XMiniAODv2,run2_nanoAOD_102Xv1:
    modifier.toModify(slimmedElectronsUpdated, src = cms.InputTag("slimmedElectronsTo106X"))


electron_id_modules_WorkingPoints_nanoAOD = cms.PSet(
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

for modifier in run2_miniAOD_80XLegacy,run2_nanoAOD_94X2016:
    modifier.toModify(electron_id_modules_WorkingPoints_nanoAOD,
        modules = cms.vstring(
            'RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Fall17_94X_V1_cff',
            'RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Fall17_94X_V2_cff',
            'RecoEgamma.ElectronIdentification.Identification.heepElectronID_HEEPV70_cff',
            'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_iso_V1_cff',
            'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_noIso_V1_cff',
            'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_iso_V2_cff',
            'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_noIso_V2_cff', 
            'RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Summer16_80X_V1_cff',
            'RecoEgamma.ElectronIdentification.Identification.cutBasedElectronHLTPreselecition_Summer16_V1_cff',
            'RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Spring15_25ns_V1_cff',
            'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring16_GeneralPurpose_V1_cff',
            'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring16_HZZ_V1_cff',
            ),
        WorkingPoints = cms.vstring(
            "egmGsfElectronIDs:cutBasedElectronID-Fall17-94X-V2-veto",
            "egmGsfElectronIDs:cutBasedElectronID-Fall17-94X-V2-loose",
            "egmGsfElectronIDs:cutBasedElectronID-Fall17-94X-V2-medium",
            "egmGsfElectronIDs:cutBasedElectronID-Fall17-94X-V2-tight",
            )
)
 

def _get_bitmapVIDForEle_docstring(modules,WorkingPoints):
    docstring=''
    for modname in modules:
        ids= __import__(modname, globals(), locals(), ['idName','cutFlow'])
        for name in dir(ids):
            _id = getattr(ids,name)
            if hasattr(_id,'idName') and hasattr(_id,'cutFlow'):
                if (len(WorkingPoints)>0 and _id.idName==WorkingPoints[0].split(':')[-1]):
                    docstring = 'VID compressed bitmap (%s), %d bits per cut'%(','.join([cut.cutName.value() for cut in _id.cutFlow]),int(ceil(log(len(WorkingPoints)+1,2))))
    return docstring

bitmapVIDForEle = cms.EDProducer("EleVIDNestedWPBitmapProducer",
    src = cms.InputTag("slimmedElectrons"),
    WorkingPoints = electron_id_modules_WorkingPoints_nanoAOD.WorkingPoints,
)
_bitmapVIDForEle_docstring = _get_bitmapVIDForEle_docstring(electron_id_modules_WorkingPoints_nanoAOD.modules,bitmapVIDForEle.WorkingPoints)

bitmapVIDForEleSpring15 = bitmapVIDForEle.clone()
bitmapVIDForEleSpring15.WorkingPoints = cms.vstring(
    "egmGsfElectronIDs:cutBasedElectronID-Spring15-25ns-V1-standalone-veto",
    "egmGsfElectronIDs:cutBasedElectronID-Spring15-25ns-V1-standalone-loose",
    "egmGsfElectronIDs:cutBasedElectronID-Spring15-25ns-V1-standalone-medium",
#    "egmGsfElectronIDs:cutBasedElectronID-Spring15-25ns-V1-standalone-tight", # not fitting in sizeof(int)
)
_bitmapVIDForEleSpring15_docstring = _get_bitmapVIDForEle_docstring(electron_id_modules_WorkingPoints_nanoAOD.modules,bitmapVIDForEleSpring15.WorkingPoints)

bitmapVIDForEleSum16 = bitmapVIDForEle.clone()
bitmapVIDForEleSum16.WorkingPoints = cms.vstring(
        "egmGsfElectronIDs:cutBasedElectronID-Summer16-80X-V1-veto",
        "egmGsfElectronIDs:cutBasedElectronID-Summer16-80X-V1-loose",
        "egmGsfElectronIDs:cutBasedElectronID-Summer16-80X-V1-medium",
        "egmGsfElectronIDs:cutBasedElectronID-Summer16-80X-V1-tight",
)
_bitmapVIDForEleSum16_docstring = _get_bitmapVIDForEle_docstring(electron_id_modules_WorkingPoints_nanoAOD.modules,bitmapVIDForEleSum16.WorkingPoints)

bitmapVIDForEleHEEP = bitmapVIDForEle.clone()
bitmapVIDForEleHEEP.WorkingPoints = cms.vstring(
    "egmGsfElectronIDs:heepElectronID-HEEPV70"
)
_bitmapVIDForEleHEEP_docstring = _get_bitmapVIDForEle_docstring(electron_id_modules_WorkingPoints_nanoAOD.modules,bitmapVIDForEleHEEP.WorkingPoints)


    

for modifier in run2_egamma_2017,run2_egamma_2018,run2_miniAOD_80XLegacy,run2_nanoAOD_94XMiniAODv1,run2_nanoAOD_94XMiniAODv2,run2_nanoAOD_94X2016,run2_nanoAOD_102Xv1,run2_nanoAOD_106Xv1:
    modifier.toModify(bitmapVIDForEle, src = "slimmedElectronsUpdated")
    modifier.toModify(bitmapVIDForEleSpring15, src = "slimmedElectronsUpdated")
    modifier.toModify(bitmapVIDForEleSum16, src = "slimmedElectronsUpdated")
    modifier.toModify(bitmapVIDForEleHEEP, src = "slimmedElectronsUpdated")


isoForEle = cms.EDProducer("EleIsoValueMapProducer",
    src = cms.InputTag("slimmedElectrons"),
    relative = cms.bool(False),
    rho_MiniIso = cms.InputTag("fixedGridRhoFastjetAll"),
    rho_PFIso = cms.InputTag("fixedGridRhoFastjetAll"),
    EAFile_MiniIso = cms.FileInPath("RecoEgamma/ElectronIdentification/data/Fall17/effAreaElectrons_cone03_pfNeuHadronsAndPhotons_94X.txt"),
    EAFile_PFIso = cms.FileInPath("RecoEgamma/ElectronIdentification/data/Fall17/effAreaElectrons_cone03_pfNeuHadronsAndPhotons_94X.txt"),
)
for modifier in run2_egamma_2017,run2_egamma_2018, run2_nanoAOD_94XMiniAODv1,run2_nanoAOD_94XMiniAODv2,run2_nanoAOD_94X2016,run2_nanoAOD_102Xv1,run2_nanoAOD_106Xv1:
    modifier.toModify(isoForEle, src = "slimmedElectronsUpdated")

run2_miniAOD_80XLegacy.toModify(isoForEle, src = "slimmedElectronsUpdated",
                                EAFile_MiniIso = "RecoEgamma/ElectronIdentification/data/Spring15/effAreaElectrons_cone03_pfNeuHadronsAndPhotons_25ns.txt",
                                EAFile_PFIso = "RecoEgamma/ElectronIdentification/data/Summer16/effAreaElectrons_cone03_pfNeuHadronsAndPhotons_80X.txt")
run2_nanoAOD_94X2016.toModify(isoForEle,
                                EAFile_MiniIso = "RecoEgamma/ElectronIdentification/data/Spring15/effAreaElectrons_cone03_pfNeuHadronsAndPhotons_25ns.txt",
                                EAFile_PFIso = "RecoEgamma/ElectronIdentification/data/Summer16/effAreaElectrons_cone03_pfNeuHadronsAndPhotons_80X.txt")

ptRatioRelForEle = cms.EDProducer("ElectronJetVarProducer",
    srcJet = cms.InputTag("updatedJets"),
    srcLep = cms.InputTag("slimmedElectrons"),
    srcVtx = cms.InputTag("offlineSlimmedPrimaryVertices"),
)
for modifier in run2_egamma_2017,run2_egamma_2018, run2_miniAOD_80XLegacy,run2_nanoAOD_94XMiniAODv1,run2_nanoAOD_94XMiniAODv2,run2_nanoAOD_94X2016,run2_nanoAOD_102Xv1,run2_nanoAOD_106Xv1:
    modifier.toModify(ptRatioRelForEle, srcLep = "slimmedElectronsUpdated")

seedGainEle = cms.EDProducer("ElectronSeedGainProducer", src = cms.InputTag("slimmedElectrons"))
for modifier in run2_egamma_2017,run2_egamma_2018, run2_miniAOD_80XLegacy,run2_nanoAOD_94XMiniAODv1,run2_nanoAOD_94XMiniAODv2,run2_nanoAOD_94X2016,run2_nanoAOD_102Xv1,run2_nanoAOD_106Xv1:
    modifier.toModify(seedGainEle, src = "slimmedElectronsUpdated")

import RecoEgamma.EgammaTools.calibratedEgammas_cff

calibratedPatElectronsUL17 = RecoEgamma.EgammaTools.calibratedEgammas_cff.calibratedPatElectrons.clone(
    produceCalibratedObjs = False,
    correctionFile = cms.string("EgammaAnalysis/ElectronTools/data/ScalesSmearings/Run2017_24Feb2020_runEtaR9Gain_v2"),
)
run2_egamma_2017.toModify(calibratedPatElectronsUL17, src = "slimmedElectronsUpdated")

calibratedPatElectronsUL18 = RecoEgamma.EgammaTools.calibratedEgammas_cff.calibratedPatElectrons.clone(
    produceCalibratedObjs = False,
    correctionFile = cms.string("EgammaAnalysis/ElectronTools/data/ScalesSmearings/Run2018_29Sep2020_RunFineEtaR9Gain"),
)
run2_egamma_2018.toModify(calibratedPatElectronsUL18, src = "slimmedElectronsUpdated")

calibratedPatElectrons80XLegacy = RecoEgamma.EgammaTools.calibratedEgammas_cff.calibratedPatElectrons.clone(
    produceCalibratedObjs = False,
    correctionFile = cms.string("EgammaAnalysis/ElectronTools/data/ScalesSmearings/Legacy2016_07Aug2017_FineEtaR9_v3_ele_unc"),
)
run2_miniAOD_80XLegacy.toModify(calibratedPatElectrons80XLegacy, src = "slimmedElectronsUpdated")

calibratedPatElectrons94X = RecoEgamma.EgammaTools.calibratedEgammas_cff.calibratedPatElectrons.clone(
    produceCalibratedObjs = False,
    correctionFile = cms.string("EgammaAnalysis/ElectronTools/data/ScalesSmearings/Run2017_17Nov2017_v1_ele_unc"),
)
for modifier in run2_nanoAOD_94XMiniAODv1,run2_nanoAOD_94XMiniAODv2,run2_nanoAOD_94X2016:
    modifier.toModify(calibratedPatElectrons94X, src = "slimmedElectronsUpdated")

calibratedPatElectrons102X = RecoEgamma.EgammaTools.calibratedEgammas_cff.calibratedPatElectrons.clone(
    produceCalibratedObjs = False,
    correctionFile = cms.string("EgammaAnalysis/ElectronTools/data/ScalesSmearings/Run2018_Step2Closure_CoarseEtaR9Gain_v2"),
)
run2_nanoAOD_102Xv1.toModify(calibratedPatElectrons102X, src = "slimmedElectronsUpdated")

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
        cutbasedID_Fall17_V2_veto = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Fall17-94X-V2-veto"),
        cutbasedID_Fall17_V2_loose = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Fall17-94X-V2-loose"),
        cutbasedID_Fall17_V2_medium = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Fall17-94X-V2-medium"),
        cutbasedID_Fall17_V2_tight = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Fall17-94X-V2-tight"),
        cutbasedID_HEEP = cms.InputTag("egmGsfElectronIDs:heepElectronID-HEEPV70"),
    ),
    userInts = cms.PSet(
        VIDNestedWPBitmap = cms.InputTag("bitmapVIDForEle"),
        VIDNestedWPBitmapHEEP = cms.InputTag("bitmapVIDForEleHEEP"),
        seedGain = cms.InputTag("seedGainEle"),
    ),
    userCands = cms.PSet(
        jetForLepJetVar = cms.InputTag("ptRatioRelForEle:jetForLepJetVar") # warning: Ptr is null if no match is found
    ),
)
for modifier in run2_egamma_2017,run2_egamma_2018, run2_miniAOD_80XLegacy,run2_nanoAOD_94XMiniAODv1,run2_nanoAOD_94XMiniAODv2,run2_nanoAOD_94X2016,run2_nanoAOD_102Xv1,run2_nanoAOD_106Xv1:
    modifier.toModify(slimmedElectronsWithUserData, src = "slimmedElectronsUpdated")


run2_egamma_2017.toModify(slimmedElectronsWithUserData.userFloats,
    ecalTrkEnergyErrPostCorrNew = cms.InputTag("calibratedPatElectronsUL17","ecalTrkEnergyErrPostCorr"),
    ecalTrkEnergyPreCorrNew     = cms.InputTag("calibratedPatElectronsUL17","ecalTrkEnergyPreCorr"),
    ecalTrkEnergyPostCorrNew    = cms.InputTag("calibratedPatElectronsUL17","ecalTrkEnergyPostCorr"),
)

run2_egamma_2018.toModify(slimmedElectronsWithUserData.userFloats,
    ecalTrkEnergyErrPostCorrNew = cms.InputTag("calibratedPatElectronsUL18","ecalTrkEnergyErrPostCorr"),
    ecalTrkEnergyPreCorrNew     = cms.InputTag("calibratedPatElectronsUL18","ecalTrkEnergyPreCorr"),
    ecalTrkEnergyPostCorrNew    = cms.InputTag("calibratedPatElectronsUL18","ecalTrkEnergyPostCorr"),
)

run2_miniAOD_80XLegacy.toModify(slimmedElectronsWithUserData.userFloats,
    mvaSpring16GP = cms.InputTag("electronMVAValueMapProducer:ElectronMVAEstimatorRun2Spring16GeneralPurposeV1Values"),
    mvaSpring16HZZ = cms.InputTag("electronMVAValueMapProducer:ElectronMVAEstimatorRun2Spring16HZZV1Values"),
)
run2_miniAOD_80XLegacy.toModify(slimmedElectronsWithUserData.userFloats,
    ecalTrkEnergyErrPostCorrNew = cms.InputTag("calibratedPatElectrons80XLegacy","ecalTrkEnergyErrPostCorr"),
    ecalTrkEnergyPreCorrNew     = cms.InputTag("calibratedPatElectrons80XLegacy","ecalTrkEnergyPreCorr"),
    ecalTrkEnergyPostCorrNew    = cms.InputTag("calibratedPatElectrons80XLegacy","ecalTrkEnergyPostCorr"),
)
run2_nanoAOD_94XMiniAODv1.toModify(slimmedElectronsWithUserData.userFloats,
    ecalTrkEnergyErrPostCorrNew = cms.InputTag("calibratedPatElectrons94X","ecalTrkEnergyErrPostCorr"),
    ecalTrkEnergyPreCorrNew     = cms.InputTag("calibratedPatElectrons94X","ecalTrkEnergyPreCorr"),
    ecalTrkEnergyPostCorrNew    = cms.InputTag("calibratedPatElectrons94X","ecalTrkEnergyPostCorr"),
)


#the94X miniAOD V2 had a bug in the scale and smearing for electrons in the E/p comb
#therefore we redo it but but we need use a new name for the userFloat as we cant override existing userfloats
#for technical reasons

run2_nanoAOD_94XMiniAODv2.toModify(slimmedElectronsWithUserData.userFloats,
    ecalTrkEnergyErrPostCorrNew = cms.InputTag("calibratedPatElectrons94X","ecalTrkEnergyErrPostCorr"),
    ecalTrkEnergyPreCorrNew     = cms.InputTag("calibratedPatElectrons94X","ecalTrkEnergyPreCorr"),
    ecalTrkEnergyPostCorrNew    = cms.InputTag("calibratedPatElectrons94X","ecalTrkEnergyPostCorr"),
)

run2_nanoAOD_102Xv1.toModify(slimmedElectronsWithUserData.userFloats,
    ecalTrkEnergyErrPostCorrNew = cms.InputTag("calibratedPatElectrons102X","ecalTrkEnergyErrPostCorr"),
    ecalTrkEnergyPreCorrNew     = cms.InputTag("calibratedPatElectrons102X","ecalTrkEnergyPreCorr"),
    ecalTrkEnergyPostCorrNew    = cms.InputTag("calibratedPatElectrons102X","ecalTrkEnergyPostCorr"),
)

run2_miniAOD_80XLegacy.toModify(slimmedElectronsWithUserData.userIntFromBools,
    mvaSpring16GP_WP90 = cms.InputTag("egmGsfElectronIDs:mvaEleID-Spring16-GeneralPurpose-V1-wp90"),
    mvaSpring16GP_WP80 = cms.InputTag("egmGsfElectronIDs:mvaEleID-Spring16-GeneralPurpose-V1-wp80"),
    mvaSpring16HZZ_WPL = cms.InputTag("egmGsfElectronIDs:mvaEleID-Spring16-HZZ-V1-wpLoose"),
    cutbasedID_Sum16_veto = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Summer16-80X-V1-veto"),
    cutbasedID_Sum16_loose = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Summer16-80X-V1-loose"),
    cutbasedID_Sum16_medium = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Summer16-80X-V1-medium"),
    cutbasedID_Sum16_tight = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Summer16-80X-V1-tight"),
    cutbasedID_HLT = cms.InputTag("egmGsfElectronIDs:cutBasedElectronHLTPreselection-Summer16-V1"),
    cutbasedID_Spring15_veto = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Spring15-25ns-V1-standalone-veto"),
    cutbasedID_Spring15_loose = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Spring15-25ns-V1-standalone-loose"),
    cutbasedID_Spring15_medium = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Spring15-25ns-V1-standalone-medium"),
    cutbasedID_Spring15_tight = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Spring15-25ns-V1-standalone-tight"),
)

run2_nanoAOD_94X2016.toModify(slimmedElectronsWithUserData.userIntFromBools,
    # MVAs and HEEP are already pre-computed. Cut-based too (except V2), but we re-add it for consistency with the nested bitmap
    cutbasedID_Sum16_veto = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Summer16-80X-V1-veto"),
    cutbasedID_Sum16_loose = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Summer16-80X-V1-loose"),
    cutbasedID_Sum16_medium = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Summer16-80X-V1-medium"),
    cutbasedID_Sum16_tight = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Summer16-80X-V1-tight"),
    cutbasedID_HLT = cms.InputTag("egmGsfElectronIDs:cutBasedElectronHLTPreselection-Summer16-V1"),
    cutbasedID_Spring15_veto = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Spring15-25ns-V1-standalone-veto"),
    cutbasedID_Spring15_loose = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Spring15-25ns-V1-standalone-loose"),
    cutbasedID_Spring15_medium = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Spring15-25ns-V1-standalone-medium"),
    cutbasedID_Spring15_tight = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Spring15-25ns-V1-standalone-tight"), 
    cutbasedID_Fall17_V2_veto = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Fall17-94X-V2-veto"),
    cutbasedID_Fall17_V2_loose = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Fall17-94X-V2-loose"),
    cutbasedID_Fall17_V2_medium = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Fall17-94X-V2-medium"),
    cutbasedID_Fall17_V2_tight = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Fall17-94X-V2-tight"),
    
)
for modifier in run2_miniAOD_80XLegacy, run2_nanoAOD_94X2016:
    modifier.toModify(slimmedElectronsWithUserData.userInts,
                      VIDNestedWPBitmapSpring15 = cms.InputTag("bitmapVIDForEleSpring15"),
                      VIDNestedWPBitmapSum16 = cms.InputTag("bitmapVIDForEleSum16"),
                      )

finalElectrons = cms.EDFilter("PATElectronRefSelector",
    src = cms.InputTag("slimmedElectronsWithUserData"),
    cut = cms.string("pt > 5 ")
)

electronMVATTH= cms.EDProducer("EleBaseMVAValueMapProducer",
    src = cms.InputTag("linkedObjects","electrons"),
    weightFile =  cms.FileInPath("PhysicsTools/NanoAOD/data/el_BDTG_2017.weights.xml"),
    name = cms.string("electronMVATTH"),
    isClassifier = cms.bool(True),
    variablesOrder = cms.vstring(["LepGood_pt","LepGood_eta","LepGood_jetNDauChargedMVASel","LepGood_miniRelIsoCharged","LepGood_miniRelIsoNeutral","LepGood_jetPtRelv2","LepGood_jetDF","LepGood_jetPtRatio","LepGood_dxy","LepGood_sip3d","LepGood_dz","LepGood_mvaFall17V2noIso"]),
    variables = cms.PSet(
        LepGood_pt = cms.string("pt"),
        LepGood_eta = cms.string("eta"),
        LepGood_jetNDauChargedMVASel = cms.string("?userCand('jetForLepJetVar').isNonnull()?userFloat('jetNDauChargedMVASel'):0"),
        LepGood_miniRelIsoCharged = cms.string("userFloat('miniIsoChg')/pt"),
        LepGood_miniRelIsoNeutral = cms.string("(userFloat('miniIsoAll')-userFloat('miniIsoChg'))/pt"),
        LepGood_jetPtRelv2 = cms.string("?userCand('jetForLepJetVar').isNonnull()?userFloat('ptRel'):0"),
        LepGood_jetDF = cms.string("?userCand('jetForLepJetVar').isNonnull()?max(userCand('jetForLepJetVar').bDiscriminator('pfDeepFlavourJetTags:probbb')+userCand('jetForLepJetVar').bDiscriminator('pfDeepFlavourJetTags:probb')+userCand('jetForLepJetVar').bDiscriminator('pfDeepFlavourJetTags:problepb'),0.0):0.0"),
        LepGood_jetPtRatio = cms.string("?userCand('jetForLepJetVar').isNonnull()?min(userFloat('ptRatio'),1.5):1.0/(1.0+userFloat('PFIsoAll04')/pt)"),
        LepGood_dxy = cms.string("log(abs(dB('PV2D')))"),
        LepGood_sip3d = cms.string("abs(dB('PV3D')/edB('PV3D'))"),
        LepGood_dz = cms.string("log(abs(dB('PVDZ')))"),
        LepGood_mvaFall17V2noIso = cms.string("userFloat('mvaFall17V2noIso')"),
    )
)
run2_egamma_2016.toModify(electronMVATTH,
    weightFile = "PhysicsTools/NanoAOD/data/el_BDTG_2016.weights.xml",
)

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
        scEtOverPt = Var("(superCluster().energy()/(pt*cosh(superCluster().eta())))-1",float,doc="(supercluster transverse energy)/pt-1",precision=8),

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

        cutBased = Var("userInt('cutbasedID_Fall17_V2_veto')+userInt('cutbasedID_Fall17_V2_loose')+userInt('cutbasedID_Fall17_V2_medium')+userInt('cutbasedID_Fall17_V2_tight')",int,doc="cut-based ID Fall17 V2 (0:fail, 1:veto, 2:loose, 3:medium, 4:tight)"),
        cutBased_Fall17_V1 = Var("userInt('cutbasedID_Fall17_V1_veto')+userInt('cutbasedID_Fall17_V1_loose')+userInt('cutbasedID_Fall17_V1_medium')+userInt('cutbasedID_Fall17_V1_tight')",int,doc="cut-based ID Fall17 V1 (0:fail, 1:veto, 2:loose, 3:medium, 4:tight)"),
        vidNestedWPBitmap = Var("userInt('VIDNestedWPBitmap')",int,doc=_bitmapVIDForEle_docstring),
        vidNestedWPBitmapHEEP = Var("userInt('VIDNestedWPBitmapHEEP')",int,doc=_bitmapVIDForEleHEEP_docstring),
        cutBased_HEEP = Var("userInt('cutbasedID_HEEP')",bool,doc="cut-based HEEP ID"),
        miniPFRelIso_chg = Var("userFloat('miniIsoChg')/pt",float,doc="mini PF relative isolation, charged component"),
        miniPFRelIso_all = Var("userFloat('miniIsoAll')/pt",float,doc="mini PF relative isolation, total (with scaled rho*EA PU corrections)"),
        pfRelIso03_chg = Var("userFloat('PFIsoChg')/pt",float,doc="PF relative isolation dR=0.3, charged component"),
        pfRelIso03_all = Var("userFloat('PFIsoAll')/pt",float,doc="PF relative isolation dR=0.3, total (with rho*EA PU corrections)"),
        jetRelIso = Var("?userCand('jetForLepJetVar').isNonnull()?(1./userFloat('ptRatio'))-1.:userFloat('PFIsoAll04')/pt",float,doc="Relative isolation in matched jet (1/ptRatio-1, pfRelIso04_all if no matched jet)",precision=8),
        jetPtRelv2 = Var("?userCand('jetForLepJetVar').isNonnull()?userFloat('ptRel'):0",float,doc="Relative momentum of the lepton with respect to the closest jet after subtracting the lepton",precision=8),
        dr03TkSumPt = Var("?pt>35?dr03TkSumPt():0",float,doc="Non-PF track isolation within a delta R cone of 0.3 with electron pt > 35 GeV",precision=8),
        dr03TkSumPtHEEP = Var("?pt>35?dr03TkSumPtHEEP():0",float,doc="Non-PF track isolation within a delta R cone of 0.3 with electron pt > 35 GeV used in HEEP ID",precision=8),
        dr03EcalRecHitSumEt = Var("?pt>35?dr03EcalRecHitSumEt():0",float,doc="Non-PF Ecal isolation within a delta R cone of 0.3 with electron pt > 35 GeV",precision=8),
        dr03HcalDepth1TowerSumEt = Var("?pt>35?dr03HcalDepth1TowerSumEt():0",float,doc="Non-PF Hcal isolation within a delta R cone of 0.3 with electron pt > 35 GeV",precision=8),
        hoe = Var("hadronicOverEm()",float,doc="H over E",precision=8),
        tightCharge = Var("isGsfCtfScPixChargeConsistent() + isGsfScPixChargeConsistent()",int,doc="Tight charge criteria (0:none, 1:isGsfScPixChargeConsistent, 2:isGsfCtfScPixChargeConsistent)"),
        convVeto = Var("passConversionVeto()",bool,doc="pass conversion veto"),
        lostHits = Var("gsfTrack.hitPattern.numberOfLostHits('MISSING_INNER_HITS')","uint8",doc="number of missing inner hits"),
        isPFcand = Var("pfCandidateRef().isNonnull()",bool,doc="electron is PF candidate"),
        seedGain = Var("userInt('seedGain')","uint8",doc="Gain of the seed crystal"),
        jetNDauCharged = Var("?userCand('jetForLepJetVar').isNonnull()?userFloat('jetNDauChargedMVASel'):0", "uint8", doc="number of charged daughters of the closest jet"),
    ),
    externalVariables = cms.PSet(
        mvaTTH = ExtVar(cms.InputTag("electronMVATTH"),float, doc="TTH MVA lepton ID score",precision=14),
    ),
)

#the94X miniAOD V2 had a bug in the scale and smearing for electrons in the E/p comb
#therefore we redo it but but we need use a new name for the userFloat as we cant override existing userfloats
#for technical reasons
for modifier in run2_egamma_2017,run2_egamma_2018, run2_nanoAOD_94XMiniAODv1,run2_nanoAOD_94XMiniAODv2,run2_miniAOD_80XLegacy,run2_nanoAOD_102Xv1:
    modifier.toModify(electronTable.variables,            
        pt = Var("pt*userFloat('ecalTrkEnergyPostCorrNew')/userFloat('ecalTrkEnergyPreCorrNew')", float, precision=-1, doc="p_{T}"),
        energyErr = Var("userFloat('ecalTrkEnergyErrPostCorrNew')", float, precision=6, doc="energy error of the cluster-track combination"),
        eCorr = Var("userFloat('ecalTrkEnergyPostCorrNew')/userFloat('ecalTrkEnergyPreCorrNew')", float, doc="ratio of the calibrated energy/miniaod energy"),
        scEtOverPt = Var("(superCluster().energy()/(pt*userFloat('ecalTrkEnergyPostCorrNew')/userFloat('ecalTrkEnergyPreCorrNew')*cosh(superCluster().eta())))-1",float,doc="(supercluster transverse energy)/pt-1",precision=8),
)

# scale and smearing only when available
for modifier in run2_nanoAOD_94X2016,:
    modifier.toModify(electronTable.variables,
        pt = Var("pt*userFloat('ecalTrkEnergyPostCorr')/userFloat('ecalTrkEnergyPreCorr')", float, precision=-1, doc="p_{T}"),
        energyErr = Var("userFloat('ecalTrkEnergyErrPostCorr')", float, precision=6, doc="energy error of the cluster-track combination"),
        eCorr = Var("userFloat('ecalTrkEnergyPostCorr')/userFloat('ecalTrkEnergyPreCorr')", float, doc="ratio of the calibrated energy/miniaod energy"),
        scEtOverPt = Var("(superCluster().energy()/(pt*userFloat('ecalTrkEnergyPostCorr')/userFloat('ecalTrkEnergyPreCorr')*cosh(superCluster().eta())))-1",float,doc="(supercluster transverse energy)/pt-1",precision=8),
    )

run2_nanoAOD_94X2016.toModify(electronTable.variables,
    cutBased_Sum16 = Var("userInt('cutbasedID_Sum16_veto')+userInt('cutbasedID_Sum16_loose')+userInt('cutbasedID_Sum16_medium')+userInt('cutbasedID_Sum16_tight')",int,doc="cut-based Summer16 ID (0:fail, 1:veto, 2:loose, 3:medium, 4:tight)"),
    cutBased_Fall17_V1 = Var("electronID('cutBasedElectronID-Fall17-94X-V1-veto')+electronID('cutBasedElectronID-Fall17-94X-V1-loose')+electronID('cutBasedElectronID-Fall17-94X-V1-medium')+electronID('cutBasedElectronID-Fall17-94X-V1-tight')",int,doc="cut-based Fall17 ID (0:fail, 1:veto, 2:loose, 3:medium, 4:tight)"),
    #cutBased in 2016 corresponds to Spring16 not Fall17V2, so have to add in V2 ID explicitly
    #it also doesnt exist in the miniAOD so have to redo it
    cutBased = Var("userInt('cutbasedID_Fall17_V2_veto')+userInt('cutbasedID_Fall17_V2_loose')+userInt('cutbasedID_Fall17_V2_medium')+userInt('cutbasedID_Fall17_V2_tight')",int,doc="cut-based ID Fall17 V2 (0:fail, 1:veto, 2:loose, 3:medium, 4:tight)"),
    cutBased_HLTPreSel = Var("userInt('cutbasedID_HLT')",int,doc="cut-based HLT pre-selection ID"),
    cutBased_HEEP = Var("electronID('heepElectronID-HEEPV70')",bool,doc="cut-based HEEP ID"),
    cutBased_Spring15 = Var("userInt('cutbasedID_Spring15_veto')+userInt('cutbasedID_Spring15_loose')+userInt('cutbasedID_Spring15_medium')+userInt('cutbasedID_Spring15_tight')",int,doc="cut-based Spring15 ID (0:fail, 1:veto, 2:loose, 3:medium, 4:tight)"),
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
    vidNestedWPBitmapSpring15 = Var("userInt('VIDNestedWPBitmapSpring15')",int,doc=_bitmapVIDForEleSpring15_docstring),
    vidNestedWPBitmapSum16 = Var("userInt('VIDNestedWPBitmapSum16')",int,doc=_bitmapVIDForEleSum16_docstring),
)
run2_miniAOD_80XLegacy.toModify(electronTable.variables,
    cutBased_Sum16 = Var("userInt('cutbasedID_Sum16_veto')+userInt('cutbasedID_Sum16_loose')+userInt('cutbasedID_Sum16_medium')+userInt('cutbasedID_Sum16_tight')",int,doc="cut-based Summer16 ID (0:fail, 1:veto, 2:loose, 3:medium, 4:tight)"),
    cutBased_HLTPreSel = Var("userInt('cutbasedID_HLT')",int,doc="cut-based HLT pre-selection ID"),
    cutBased_Spring15 = Var("userInt('cutbasedID_Spring15_veto')+userInt('cutbasedID_Spring15_loose')+userInt('cutbasedID_Spring15_medium')+userInt('cutbasedID_Spring15_tight')",int,doc="cut-based Spring15 ID (0:fail, 1:veto, 2:loose, 3:medium, 4:tight)"),
    mvaSpring16GP = Var("userFloat('mvaSpring16GP')",float,doc="MVA general-purpose ID score"),
    mvaSpring16GP_WP80 = Var("userInt('mvaSpring16GP_WP80')",bool,doc="MVA general-purpose ID WP80"),
    mvaSpring16GP_WP90 = Var("userInt('mvaSpring16GP_WP90')",bool,doc="MVA general-purpose ID WP90"),
    mvaSpring16HZZ = Var("userFloat('mvaSpring16HZZ')",float,doc="MVA HZZ ID score"),
    mvaSpring16HZZ_WPL = Var("userInt('mvaSpring16HZZ_WPL')",bool,doc="MVA HZZ ID loose WP"),

    vidNestedWPBitmapSpring15 = Var("userInt('VIDNestedWPBitmapSpring15')",int,doc=_bitmapVIDForEleSpring15_docstring),
    vidNestedWPBitmapSum16 = Var("userInt('VIDNestedWPBitmapSum16')",int,doc=_bitmapVIDForEleSum16_docstring),

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

electronSequence = cms.Sequence(bitmapVIDForEle + bitmapVIDForEleHEEP + isoForEle + ptRatioRelForEle + seedGainEle + slimmedElectronsWithUserData + finalElectrons)
electronTables = cms.Sequence (electronMVATTH + electronTable)
electronMC = cms.Sequence(electronsMCMatchForTable + electronMCTable)
from RecoEgamma.ElectronIdentification.heepIdVarValueMapProducer_cfi import heepIDVarValueMaps
_updateTo106X_sequence =cms.Sequence(heepIDVarValueMaps + slimmedElectronsTo106X)
heepIDVarValueMaps.dataFormat = 2

_withTo106XAndUpdate_sequence = cms.Sequence(_updateTo106X_sequence + slimmedElectronsUpdated + electronSequence.copy())

_withULAndUpdate_sequence = cms.Sequence(slimmedElectronsUpdated + electronSequence.copy())

_withUL17AndUpdateScale_sequence = _withULAndUpdate_sequence.copy()
_withUL17AndUpdateScale_sequence.replace(slimmedElectronsWithUserData, calibratedPatElectronsUL17 + slimmedElectronsWithUserData)
run2_egamma_2017.toReplaceWith(electronSequence, _withUL17AndUpdateScale_sequence)

_withUL18AndUpdateScale_sequence = _withULAndUpdate_sequence.copy()
_withUL18AndUpdateScale_sequence.replace(slimmedElectronsWithUserData, calibratedPatElectronsUL18 + slimmedElectronsWithUserData)
run2_egamma_2018.toReplaceWith(electronSequence, _withUL18AndUpdateScale_sequence)

_withTo106XAndUpdateAnd80XLegacyScale_sequence = _withTo106XAndUpdate_sequence.copy()
_withTo106XAndUpdateAnd80XLegacyScale_sequence.replace(slimmedElectronsWithUserData, calibratedPatElectrons80XLegacy + bitmapVIDForEleSpring15 +bitmapVIDForEleSum16 + slimmedElectronsWithUserData)
run2_miniAOD_80XLegacy.toReplaceWith(electronSequence, _withTo106XAndUpdateAnd80XLegacyScale_sequence)

_withTo106XAndUpdateAnd94XScale_sequence = _withTo106XAndUpdate_sequence.copy()
_withTo106XAndUpdateAnd94XScale_sequence.replace(slimmedElectronsWithUserData, calibratedPatElectrons94X + slimmedElectronsWithUserData)
run2_nanoAOD_94XMiniAODv1.toReplaceWith(electronSequence, _withTo106XAndUpdateAnd94XScale_sequence)
run2_nanoAOD_94XMiniAODv2.toReplaceWith(electronSequence, _withTo106XAndUpdateAnd94XScale_sequence)

_withTo106XAndUpdateAnd_bitmapVIDForEleSpring15AndSum16_sequence = _withTo106XAndUpdate_sequence.copy()
_withTo106XAndUpdateAnd_bitmapVIDForEleSpring15AndSum16_sequence.replace(slimmedElectronsWithUserData,  bitmapVIDForEleSpring15 + bitmapVIDForEleSum16 + slimmedElectronsWithUserData)
run2_nanoAOD_94X2016.toReplaceWith(electronSequence, _withTo106XAndUpdateAnd_bitmapVIDForEleSpring15AndSum16_sequence)

_withTo106XAndUpdateAnd102XScale_sequence = _withTo106XAndUpdate_sequence.copy()
_withTo106XAndUpdateAnd102XScale_sequence.replace(slimmedElectronsWithUserData, calibratedPatElectrons102X + slimmedElectronsWithUserData)
run2_nanoAOD_102Xv1.toReplaceWith(electronSequence, _withTo106XAndUpdateAnd102XScale_sequence)

_withUpdate_sequence = electronSequence.copy()
_withUpdate_sequence.replace(bitmapVIDForEle, slimmedElectronsUpdated + bitmapVIDForEle)
run2_nanoAOD_106Xv1.toReplaceWith(electronSequence, _withUpdate_sequence)
