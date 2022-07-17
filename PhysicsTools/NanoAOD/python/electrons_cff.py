import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.nano_eras_cff import *
from PhysicsTools.NanoAOD.common_cff import *
import PhysicsTools.PatAlgos.producersLayer1.electronProducer_cfi
from math import ceil,log
#NOTE: All definitions of modules should point to the latest flavour of the electronTable in NanoAOD.
#Common modifications for past eras are done at the end whereas modifications specific to a single era is done after the original definition.
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
    src = cms.InputTag("slimmedElectrons"),
    vertices = cms.InputTag("offlineSlimmedPrimaryVertices"),
    computeMiniIso = cms.bool(False),
    fixDxySign = cms.bool(True),
    pfCandsForMiniIso = cms.InputTag("packedPFCandidates"),
    miniIsoParamsB = PhysicsTools.PatAlgos.producersLayer1.electronProducer_cfi.patElectrons.miniIsoParamsB, # so they're in sync
    miniIsoParamsE = PhysicsTools.PatAlgos.producersLayer1.electronProducer_cfi.patElectrons.miniIsoParamsE, # so they're in sync
)
run2_miniAOD_80XLegacy.toModify( slimmedElectronsUpdated, computeMiniIso = True )
##modify the past eras
for modifier in run2_miniAOD_80XLegacy,run2_nanoAOD_94X2016,run2_nanoAOD_94XMiniAODv1,run2_nanoAOD_94XMiniAODv2,run2_nanoAOD_102Xv1:
    modifier.toModify(slimmedElectronsUpdated, src = cms.InputTag("slimmedElectronsTo106X"))
############################FOR bitmapVIDForEle main defn#############################
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

bitmapVIDForEleSpring15 = bitmapVIDForEle.clone(
    WorkingPoints = cms.vstring(
        "egmGsfElectronIDs:cutBasedElectronID-Spring15-25ns-V1-standalone-veto",
        "egmGsfElectronIDs:cutBasedElectronID-Spring15-25ns-V1-standalone-loose",
        "egmGsfElectronIDs:cutBasedElectronID-Spring15-25ns-V1-standalone-medium",
        #    "egmGsfElectronIDs:cutBasedElectronID-Spring15-25ns-V1-standalone-tight", # not fitting in sizeof(int)
    )
)
_bitmapVIDForEleSpring15_docstring = _get_bitmapVIDForEle_docstring(electron_id_modules_WorkingPoints_nanoAOD.modules,bitmapVIDForEleSpring15.WorkingPoints)

bitmapVIDForEleSum16 = bitmapVIDForEle.clone(
    WorkingPoints = cms.vstring(
        "egmGsfElectronIDs:cutBasedElectronID-Summer16-80X-V1-veto",
        "egmGsfElectronIDs:cutBasedElectronID-Summer16-80X-V1-loose",
        "egmGsfElectronIDs:cutBasedElectronID-Summer16-80X-V1-medium",
        "egmGsfElectronIDs:cutBasedElectronID-Summer16-80X-V1-tight",
    )
)
_bitmapVIDForEleSum16_docstring = _get_bitmapVIDForEle_docstring(electron_id_modules_WorkingPoints_nanoAOD.modules,bitmapVIDForEleSum16.WorkingPoints)

bitmapVIDForEleHEEP = bitmapVIDForEle.clone(
    WorkingPoints = cms.vstring("egmGsfElectronIDs:heepElectronID-HEEPV70"
    )
)
_bitmapVIDForEleHEEP_docstring = _get_bitmapVIDForEle_docstring(electron_id_modules_WorkingPoints_nanoAOD.modules,bitmapVIDForEleHEEP.WorkingPoints)
############################for bitmapVIDForEle defn end#############################
#######################ISO ELE defn(in principle should be an import####################
##PhysicsTools/NanoAOD/python/EleIsoValueMapProducer_cfi.py
isoForEle = cms.EDProducer("EleIsoValueMapProducer",
    src = cms.InputTag("slimmedElectrons"),
    relative = cms.bool(False),
    rho_MiniIso = cms.InputTag("fixedGridRhoFastjetAll"),
    rho_PFIso = cms.InputTag("fixedGridRhoFastjetAll"),
    EAFile_MiniIso = cms.FileInPath("RecoEgamma/ElectronIdentification/data/Fall17/effAreaElectrons_cone03_pfNeuHadronsAndPhotons_94X.txt"),
    EAFile_PFIso = cms.FileInPath("RecoEgamma/ElectronIdentification/data/Fall17/effAreaElectrons_cone03_pfNeuHadronsAndPhotons_94X.txt"),
)
run2_miniAOD_80XLegacy.toModify(isoForEle,
                                EAFile_MiniIso = "RecoEgamma/ElectronIdentification/data/Spring15/effAreaElectrons_cone03_pfNeuHadronsAndPhotons_25ns.txt",
                                EAFile_PFIso = "RecoEgamma/ElectronIdentification/data/Summer16/effAreaElectrons_cone03_pfNeuHadronsAndPhotons_80X.txt")
run2_nanoAOD_94X2016.toModify(isoForEle,
                                EAFile_MiniIso = "RecoEgamma/ElectronIdentification/data/Spring15/effAreaElectrons_cone03_pfNeuHadronsAndPhotons_25ns.txt",
                                EAFile_PFIso = "RecoEgamma/ElectronIdentification/data/Summer16/effAreaElectrons_cone03_pfNeuHadronsAndPhotons_80X.txt")
#######################################ISO ELE end#####################################
######################################ptRatioForEle#####################################
###import from hysicsTools/NanoAOD/pythonElectronJetVarProducer_cfi.py
ptRatioRelForEle = cms.EDProducer("ElectronJetVarProducer",
    srcJet = cms.InputTag("updatedJetsPuppi"),
    srcLep = cms.InputTag("slimmedElectrons"),
    srcVtx = cms.InputTag("offlineSlimmedPrimaryVertices"),
)
######################################ptRatioForEle#####################################
#############3###################seedGailEle#############################
seedGainEle = cms.EDProducer("ElectronSeedGainProducer", src = cms.InputTag("slimmedElectrons"))
############################################seed gainELE
############################calibratedPatElectrons##############
##this is a special one, so we leave the era modifications here#####
import RecoEgamma.EgammaTools.calibratedEgammas_cff

calibratedPatElectronsNano = RecoEgamma.EgammaTools.calibratedEgammas_cff.calibratedPatElectrons.clone(
    produceCalibratedObjs = False,
    src = "slimmedElectrons"
)

(run2_egamma_2016 & tracker_apv_vfp30_2016).toModify(calibratedPatElectronsNano,
    correctionFile = cms.string("EgammaAnalysis/ElectronTools/data/ScalesSmearings/Run2016_UltraLegacy_preVFP_RunFineEtaR9Gain")
) 

(run2_egamma_2016 & ~tracker_apv_vfp30_2016).toModify(calibratedPatElectronsNano,
    correctionFile = cms.string("EgammaAnalysis/ElectronTools/data/ScalesSmearings/Run2016_UltraLegacy_postVFP_RunFineEtaR9Gain")
)

run2_egamma_2017.toModify(calibratedPatElectronsNano,
    correctionFile = cms.string("EgammaAnalysis/ElectronTools/data/ScalesSmearings/Run2017_24Feb2020_runEtaR9Gain_v2")
)

run2_egamma_2018.toModify(calibratedPatElectronsNano,
    correctionFile = cms.string("EgammaAnalysis/ElectronTools/data/ScalesSmearings/Run2018_29Sep2020_RunFineEtaR9Gain")
)

run2_miniAOD_80XLegacy.toModify(calibratedPatElectronsNano,
                                correctionFile = cms.string("EgammaAnalysis/ElectronTools/data/ScalesSmearings/Legacy2016_07Aug2017_FineEtaR9_v3_ele_unc")
                            )

for modifier in run2_nanoAOD_94XMiniAODv1,run2_nanoAOD_94XMiniAODv2:
    modifier.toModify(calibratedPatElectronsNano, 
        correctionFile = cms.string("EgammaAnalysis/ElectronTools/data/ScalesSmearings/Run2017_17Nov2017_v1_ele_unc")
    )

run2_nanoAOD_102Xv1.toModify(calibratedPatElectronsNano,
    correctionFile = cms.string("EgammaAnalysis/ElectronTools/data/ScalesSmearings/Run2018_Step2Closure_CoarseEtaR9Gain_v2")
)
##############################end calibratedPatElectronsNano############################33
#####################Start slimmedElectronsWithUserData###############################3
##import from PhysicsTools/PatAlgos/python/electronsWithUserData_cfi.py
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
        ecalTrkEnergyErrPostCorrNew = cms.InputTag("calibratedPatElectronsNano","ecalTrkEnergyErrPostCorr"),
        ecalTrkEnergyPreCorrNew     = cms.InputTag("calibratedPatElectronsNano","ecalTrkEnergyPreCorr"),
        ecalTrkEnergyPostCorrNew    = cms.InputTag("calibratedPatElectronsNano","ecalTrkEnergyPostCorr"),
        energyScaleUpNew               = cms.InputTag("calibratedPatElectronsNano","energyScaleUp"),
        energyScaleDownNew             = cms.InputTag("calibratedPatElectronsNano","energyScaleDown"),
        energySigmaUpNew               = cms.InputTag("calibratedPatElectronsNano","energySigmaUp"),
        energySigmaDownNew             = cms.InputTag("calibratedPatElectronsNano","energySigmaDown"),

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

###Not to update with S+S vars as they already exist for run2_nanoAOD_94X2016 era
run2_nanoAOD_94X2016.toModify(slimmedElectronsWithUserData.userFloats,
                              ecalTrkEnergyErrPostCorrNew = None,
                              ecalTrkEnergyPreCorrNew = None,
                              ecalTrkEnergyPostCorrNew = None,
                              energyScaleUpNew               = None,
                              energyScaleDownNew             = None,
                              energySigmaUpNew               = None,
                              energySigmaDownNew             = None
                              
                              
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

run2_miniAOD_80XLegacy.toModify(slimmedElectronsWithUserData.userFloats,
    mvaSpring16GP = cms.InputTag("electronMVAValueMapProducer:ElectronMVAEstimatorRun2Spring16GeneralPurposeV1Values"),
    mvaSpring16HZZ = cms.InputTag("electronMVAValueMapProducer:ElectronMVAEstimatorRun2Spring16HZZV1Values"),
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

for modifier in run2_miniAOD_80XLegacy, run2_nanoAOD_94X2016:
    modifier.toModify(slimmedElectronsWithUserData.userInts,
                      VIDNestedWPBitmapSpring15 = cms.InputTag("bitmapVIDForEleSpring15"),
                      VIDNestedWPBitmapSum16 = cms.InputTag("bitmapVIDForEleSum16"),
                      )
#################################################END slimmedElectrons with user data#####################
#################################################finalElectrons#####################
finalElectrons = cms.EDFilter("PATElectronRefSelector",
    src = cms.InputTag("slimmedElectronsWithUserData"),
    cut = cms.string("pt > 5 ")
)
#################################################finalElectrons#####################
################################################electronMVATTH#####################
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
################################################electronMVATTH end#####################
################################################electronTable defn #####################
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

        mvaFall17V2Iso = Var("userFloat('mvaFall17V2Iso')",float,doc="MVA Iso ID V2 score"),
        mvaFall17V2Iso_WP80 = Var("userInt('mvaFall17V2Iso_WP80')",bool,doc="MVA Iso ID V2 WP80"),
        mvaFall17V2Iso_WP90 = Var("userInt('mvaFall17V2Iso_WP90')",bool,doc="MVA Iso ID V2 WP90"),
        mvaFall17V2Iso_WPL = Var("userInt('mvaFall17V2Iso_WPL')",bool,doc="MVA Iso ID V2 loose WP"),
        mvaFall17V2noIso = Var("userFloat('mvaFall17V2noIso')",float,doc="MVA noIso ID V2 score"),
        mvaFall17V2noIso_WP80 = Var("userInt('mvaFall17V2noIso_WP80')",bool,doc="MVA noIso ID V2 WP80"),
        mvaFall17V2noIso_WP90 = Var("userInt('mvaFall17V2noIso_WP90')",bool,doc="MVA noIso ID V2 WP90"),
        mvaFall17V2noIso_WPL = Var("userInt('mvaFall17V2noIso_WPL')",bool,doc="MVA noIso ID V2 loose WP"),

        cutBased = Var("userInt('cutbasedID_Fall17_V2_veto')+userInt('cutbasedID_Fall17_V2_loose')+userInt('cutbasedID_Fall17_V2_medium')+userInt('cutbasedID_Fall17_V2_tight')",int,doc="cut-based ID Fall17 V2 (0:fail, 1:veto, 2:loose, 3:medium, 4:tight)"),
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
        dr03HcalDepth1TowerSumEt = Var("?pt>35?dr03HcalTowerSumEt(1):0",float,doc="Non-PF Hcal isolation within a delta R cone of 0.3 with electron pt > 35 GeV",precision=8),
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
        fsrPhotonIdx = ExtVar(cms.InputTag("leptonFSRphotons:eleFsrIndex"),int, doc="Index of the lowest-dR/ET2 among associated FSR photons"),
    ),
)

#for technical reasons
for modifier in run2_nanoAOD_94XMiniAODv1,run2_nanoAOD_94XMiniAODv2,run2_miniAOD_80XLegacy,run2_nanoAOD_102Xv1,run2_nanoAOD_106Xv1,run2_nanoAOD_106Xv2:
    modifier.toModify(electronTable.variables,            
        pt = Var("pt*userFloat('ecalTrkEnergyPostCorrNew')/userFloat('ecalTrkEnergyPreCorrNew')", float, precision=-1, doc="p_{T}"),
        energyErr = Var("userFloat('ecalTrkEnergyErrPostCorrNew')", float, precision=6, doc="energy error of the cluster-track combination"),
        eCorr = Var("userFloat('ecalTrkEnergyPostCorrNew')/userFloat('ecalTrkEnergyPreCorrNew')", float, doc="ratio of the calibrated energy/miniaod energy"),
        scEtOverPt = Var("(superCluster().energy()/(pt*userFloat('ecalTrkEnergyPostCorrNew')/userFloat('ecalTrkEnergyPreCorrNew')*cosh(superCluster().eta())))-1",float,doc="(supercluster transverse energy)/pt-1",precision=8),
        dEscaleUp=Var("userFloat('ecalTrkEnergyPostCorrNew')-userFloat('energyScaleUpNew')", float,  doc="ecal energy scale shifted 1 sigma up(adding gain/stat/syst in quadrature)", precision=8),
        dEscaleDown=Var("userFloat('ecalTrkEnergyPostCorrNew')-userFloat('energyScaleDownNew')", float,  doc="ecal energy scale shifted 1 sigma down (adding gain/stat/syst in quadrature)", precision=8),
        dEsigmaUp=Var("userFloat('ecalTrkEnergyPostCorrNew')-userFloat('energySigmaUpNew')", float, doc="ecal energy smearing value shifted 1 sigma up", precision=8),
        dEsigmaDown=Var("userFloat('ecalTrkEnergyPostCorrNew')-userFloat('energySigmaDownNew')", float,  doc="ecal energy smearing value shifted 1 sigma up", precision=8),

)
##Keeping the possibilty of using V1 working points in older eras
(run2_nanoAOD_92X | run2_nanoAOD_94XMiniAODv1 | run2_nanoAOD_94XMiniAODv2 | run2_nanoAOD_94X2016 | run2_nanoAOD_102Xv1).toModify(electronTable.variables,
        mvaFall17V1Iso = Var("userFloat('mvaFall17V1Iso')",float,doc="MVA Iso ID V1 score"),
        mvaFall17V1Iso_WP80 = Var("userInt('mvaFall17V1Iso_WP80')",bool,doc="MVA Iso ID V1 WP80"),
        mvaFall17V1Iso_WP90 = Var("userInt('mvaFall17V1Iso_WP90')",bool,doc="MVA Iso ID V1 WP90"),
        mvaFall17V1Iso_WPL = Var("userInt('mvaFall17V1Iso_WPL')",bool,doc="MVA Iso ID V1 loose WP"),
        mvaFall17V1noIso = Var("userFloat('mvaFall17V1noIso')",float,doc="MVA noIso ID V1 score"),
        mvaFall17V1noIso_WP80 = Var("userInt('mvaFall17V1noIso_WP80')",bool,doc="MVA noIso ID V1 WP80"),
        mvaFall17V1noIso_WP90 = Var("userInt('mvaFall17V1noIso_WP90')",bool,doc="MVA noIso ID V1 WP90"),
        mvaFall17V1noIso_WPL = Var("userInt('mvaFall17V1noIso_WPL')",bool,doc="MVA noIso ID V1 loose WP"),
        cutBased_Fall17_V1 = Var("userInt('cutbasedID_Fall17_V1_veto')+userInt('cutbasedID_Fall17_V1_loose')+userInt('cutbasedID_Fall17_V1_medium')+userInt('cutbasedID_Fall17_V1_tight')",int,doc="cut-based ID Fall17 V1 (0:fail, 1:veto, 2:loose, 3:medium, 4:tight)"),
)

#the94X miniAOD V2 had a bug in the scale and smearing for electrons in the E/p comb
#therefore we redo it but but we need use a new name for the userFloat as we cant override existing userfloats
# scale and smearing only when available#ONLY needed for this era
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
    pt = Var("pt*userFloat('ecalTrkEnergyPostCorr')/userFloat('ecalTrkEnergyPreCorr')", float, precision=-1, doc="p_{T}"),
    energyErr = Var("userFloat('ecalTrkEnergyErrPostCorr')", float, precision=6, doc="energy error of the cluster-track combination"),
    eCorr = Var("userFloat('ecalTrkEnergyPostCorr')/userFloat('ecalTrkEnergyPreCorr')", float, doc="ratio of the calibrated energy/miniaod energy"),
    scEtOverPt = Var("(superCluster().energy()/(pt*userFloat('ecalTrkEnergyPostCorr')/userFloat('ecalTrkEnergyPreCorr')*cosh(superCluster().eta())))-1",float,doc="(supercluster transverse energy)/pt-1",precision=8),
    dEscaleUp=Var("userFloat('ecalTrkEnergyPostCorr')-userFloat('energyScaleUp')", float,  doc="ecal energy scale shifted 1 sigma up (adding gain/stat/syst in quadrature)", precision=8),
    dEscaleDown=Var("userFloat('ecalTrkEnergyPostCorr')-userFloat('energyScaleDown')", float,  doc="ecal energy scale shifted 1 sigma down (adding gain/stat/syst in quadrature)", precision=8),
    dEsigmaUp=Var("userFloat('ecalTrkEnergyPostCorr')-userFloat('energySigmaUp')", float, doc="ecal energy smearing value shifted 1 sigma up", precision=8),
    dEsigmaDown=Var("userFloat('ecalTrkEnergyPostCorr')-userFloat('energySigmaDown')", float,  doc="ecal energy smearing value shifted 1 sigma up", precision=8),
)
###
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
#############electron Table END#####################
# Depends on particlelevel producer run in particlelevel_cff
tautaggerForMatching = cms.EDProducer("GenJetTauTaggerProducer",
                                      src = cms.InputTag('particleLevel:leptons')
)
 ##PhysicsTools/NanoAOD/plugins/GenJetGenPartMerger.cc##this class misses fillDescription#TODO
matchingElecPhoton = cms.EDProducer("GenJetGenPartMerger",
                                    srcJet =cms.InputTag("particleLevel:leptons"),
                                    srcPart=cms.InputTag("particleLevel:photons"),
                                    cut = cms.string("pt > 3"),
                                    hasTauAnc=cms.InputTag("tautaggerForMatching"),
)
electronsMCMatchForTableAlt = cms.EDProducer("GenJetMatcherDRPtByDR",  # cut on deltaR, deltaPt/Pt; pick best by deltaR
    src         = electronTable.src,                 # final reco collection
    matched     = cms.InputTag("matchingElecPhoton:merged"), # final mc-truth particle collection
    mcPdgId     = cms.vint32(11,22),                 # one or more PDG ID (11 = el, 22 = pho); absolute values (see below)
    checkCharge = cms.bool(False),              # True = require RECO and MC objects to have the same charge
    mcStatus    = cms.vint32(),
    maxDeltaR   = cms.double(0.3),              # Minimum deltaR for the match
    maxDPtRel   = cms.double(0.5),              # Minimum deltaPt/Pt for the match
    resolveAmbiguities    = cms.bool(True),     # Forbid two RECO objects to match to the same GEN object
    resolveByMatchQuality = cms.bool(True),    # False = just match input in order; True = pick lowest deltaR pair first
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
#should be cloned from PhysicsTools/NanoAOD/python/candMcMatchTable_cfi.py 
electronMCTable = cms.EDProducer("CandMCMatchTableProducer",
    src     = electronTable.src,
    mcMapDressedLep = cms.InputTag("electronsMCMatchForTableAlt"),
    mcMap   = cms.InputTag("electronsMCMatchForTable"),
    mapTauAnc = cms.InputTag("matchingElecPhoton:hasTauAnc"),
    objName = electronTable.name,
    objType = electronTable.name, #cms.string("Electron"),
    branchName = cms.string("genPart"),
    docString = cms.string("MC matching to status==1 electrons or photons"),
    genparticles     = cms.InputTag("finalGenParticles"), 
)

electronTask = cms.Task(bitmapVIDForEle,bitmapVIDForEleHEEP,isoForEle,ptRatioRelForEle,seedGainEle,calibratedPatElectronsNano,slimmedElectronsWithUserData,finalElectrons)
electronTablesTask = cms.Task(electronMVATTH, electronTable)
electronMCTask = cms.Task(tautaggerForMatching, matchingElecPhoton, electronsMCMatchForTable, electronsMCMatchForTableAlt, electronMCTable)

## TEMPORARY as no ID for Run3 yet
(run3_nanoAOD_devel).toReplaceWith(electronTask, electronTask.copyAndExclude([bitmapVIDForEle,bitmapVIDForEleHEEP]))
(run3_nanoAOD_devel).toModify(slimmedElectronsWithUserData, userIntFromBools = cms.PSet())
(run3_nanoAOD_devel).toModify(slimmedElectronsWithUserData.userInts,
                                                    VIDNestedWPBitmap = None,
                                                    VIDNestedWPBitmapHEEP = None)
(run3_nanoAOD_devel).toModify(slimmedElectronsWithUserData.userFloats,
                                                    mvaFall17V1Iso = None,
                                                    mvaFall17V1noIso = None,
                                                    mvaFall17V2Iso = None,
                                                    mvaFall17V2noIso = None,
                                                )
(run3_nanoAOD_devel).toModify(electronTable.variables,
                              mvaFall17V2Iso = None,
                              mvaFall17V2Iso_WP80 = None,
                              mvaFall17V2Iso_WP90 = None,
                              mvaFall17V2Iso_WPL = None,
                              mvaFall17V2noIso = None,
                              mvaFall17V2noIso_WP80 = None,
                              mvaFall17V2noIso_WP90 = None,
                              mvaFall17V2noIso_WPL = None,
                              vidNestedWPBitmapHEEP = None,
                              vidNestedWPBitmap = None,
                              cutBased = None,
                              cutBased_HEEP = None,
)

(run3_nanoAOD_devel).toReplaceWith(electronTablesTask, electronTablesTask.copyAndExclude([electronMVATTH]))
(run3_nanoAOD_devel).toModify(electronTable, externalVariables = cms.PSet(fsrPhotonIdx = ExtVar(cms.InputTag("leptonFSRphotons:eleFsrIndex"),int, doc="Index of the  lowest-dR/ET2 among associated FSR photons")),
)
##### end TEMPORARY Run3

# Revert back to AK4 CHS jets for Run 2
run2_nanoAOD_ANY.toModify(ptRatioRelForEle,srcJet="updatedJets")


#for NANO from reminAOD, no need to run slimmedElectronsUpdated, other modules of electron sequence will run on slimmedElectrons
for modifier in run2_miniAOD_80XLegacy,run2_nanoAOD_94XMiniAODv1,run2_nanoAOD_94XMiniAODv2,run2_nanoAOD_94X2016,run2_nanoAOD_102Xv1,run2_nanoAOD_106Xv1:
    modifier.toModify(bitmapVIDForEle, src = "slimmedElectronsUpdated")
    modifier.toModify(bitmapVIDForEleSpring15, src = "slimmedElectronsUpdated")
    modifier.toModify(bitmapVIDForEleSum16, src = "slimmedElectronsUpdated")
    modifier.toModify(bitmapVIDForEleHEEP, src = "slimmedElectronsUpdated")
    modifier.toModify(isoForEle, src = "slimmedElectronsUpdated")
    modifier.toModify(ptRatioRelForEle, srcLep = "slimmedElectronsUpdated")
    modifier.toModify(seedGainEle, src = "slimmedElectronsUpdated")
    modifier.toModify(slimmedElectronsWithUserData, src = "slimmedElectronsUpdated")
    modifier.toModify(calibratedPatElectronsNano, src = "slimmedElectronsUpdated")
    ###this sequence should run for all eras except run2_nanoAOD_106Xv2 which should run the electronSequence as above
    _withULAndUpdate_Task = cms.Task(slimmedElectronsUpdated)
    _withULAndUpdate_Task.add(electronTask.copy())
    modifier.toReplaceWith(electronTask, _withULAndUpdate_Task)

from RecoEgamma.ElectronIdentification.heepIdVarValueMapProducer_cfi import heepIDVarValueMaps
_withTo106XAndUpdate_Task = cms.Task(heepIDVarValueMaps,slimmedElectronsTo106X)
_withTo106XAndUpdate_Task.add(electronTask.copy())
heepIDVarValueMaps.dataFormat = 2

for modifier in run2_nanoAOD_94XMiniAODv2, run2_nanoAOD_94X2016, run2_nanoAOD_102Xv1, run2_nanoAOD_94XMiniAODv1:
    modifier.toReplaceWith(electronTask, _withTo106XAndUpdate_Task)

for modifier in run2_miniAOD_80XLegacy, run2_nanoAOD_94X2016:
    _withTo106XAndUpdateAnd80XLegacyScale_Task = cms.Task(bitmapVIDForEleSpring15,bitmapVIDForEleSum16)
    _withTo106XAndUpdateAnd80XLegacyScale_Task.add(_withTo106XAndUpdate_Task.copy())
    modifier.toReplaceWith(electronTask, _withTo106XAndUpdateAnd80XLegacyScale_Task)

_withTo106XAndUpdateAnd94XScale_Task = _withTo106XAndUpdate_Task.copy()
for modifier in run2_nanoAOD_94XMiniAODv1, run2_nanoAOD_94XMiniAODv2,run2_nanoAOD_102Xv1:
    modifier.toReplaceWith(electronTask, _withTo106XAndUpdate_Task)
