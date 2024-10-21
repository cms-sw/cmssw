import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.simplePATElectronFlatTableProducer_cfi import simplePATElectronFlatTableProducer
from PhysicsTools.NanoAOD.nano_eras_cff import *
from PhysicsTools.NanoAOD.common_cff import *
from math import ceil,log

############################FOR bitmapVIDForEle main defn#############################
electron_id_modules_WorkingPoints_nanoAOD = cms.PSet(
    modules = cms.vstring(        
        'RecoEgamma.ElectronIdentification.Identification.heepElectronID_HEEPV70_cff',
        # HZZ ID
        'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Summer16UL_ID_ISO_cff',
        'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Summer17UL_ID_ISO_cff',
        'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Summer18UL_ID_ISO_cff',
        'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Winter22_HZZ_V1_cff',
        # Fall17: need to include the modules too to make sure they are run
        'RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Fall17_94X_V2_cff',
        'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_iso_V2_cff',
        'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_noIso_V2_cff',
        # Run3Winter22:
        'RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Winter22_122X_V1_cff',
        'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_RunIIIWinter22_iso_V1_cff',
        'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_RunIIIWinter22_noIso_V1_cff',
    ),
    WorkingPoints = cms.vstring(
        "egmGsfElectronIDs:cutBasedElectronID-RunIIIWinter22-V1-veto",
        "egmGsfElectronIDs:cutBasedElectronID-RunIIIWinter22-V1-loose",
        "egmGsfElectronIDs:cutBasedElectronID-RunIIIWinter22-V1-medium",
        "egmGsfElectronIDs:cutBasedElectronID-RunIIIWinter22-V1-tight",
    )
)

# Use Fall17-94X-V2 as default for Run 2
electron_id_modules_WorkingPoints_nanoAOD_Run2 = cms.PSet(
    modules = cms.vstring(
        'RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Fall17_94X_V2_cff',
        'RecoEgamma.ElectronIdentification.Identification.heepElectronID_HEEPV70_cff',
        'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_iso_V2_cff',
        'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_noIso_V2_cff',
        # HZZ ID
        'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Summer16UL_ID_ISO_cff',
        'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Summer17UL_ID_ISO_cff',
        'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Summer18UL_ID_ISO_cff',
    ),
    WorkingPoints = cms.vstring(
        "egmGsfElectronIDs:cutBasedElectronID-Fall17-94X-V2-veto",
        "egmGsfElectronIDs:cutBasedElectronID-Fall17-94X-V2-loose",
        "egmGsfElectronIDs:cutBasedElectronID-Fall17-94X-V2-medium",
        "egmGsfElectronIDs:cutBasedElectronID-Fall17-94X-V2-tight",
    )
)

# make Fall17 the default one in Run2
run2_egamma.toModify(electron_id_modules_WorkingPoints_nanoAOD,
                          modules=electron_id_modules_WorkingPoints_nanoAOD_Run2.modules).\
        toModify(electron_id_modules_WorkingPoints_nanoAOD,
                 WorkingPoints=electron_id_modules_WorkingPoints_nanoAOD_Run2.WorkingPoints)

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
    srcForID = cms.InputTag("reducedEgamma","reducedGedGsfElectrons"),
    WorkingPoints = electron_id_modules_WorkingPoints_nanoAOD.WorkingPoints,
)
_bitmapVIDForEle_docstring = _get_bitmapVIDForEle_docstring(electron_id_modules_WorkingPoints_nanoAOD.modules,bitmapVIDForEle.WorkingPoints)

bitmapVIDForEleFall17V2 = bitmapVIDForEle.clone(
    WorkingPoints = electron_id_modules_WorkingPoints_nanoAOD_Run2.WorkingPoints
    )
_bitmapVIDForEleFall17V2_docstring = _get_bitmapVIDForEle_docstring(electron_id_modules_WorkingPoints_nanoAOD.modules, bitmapVIDForEleFall17V2.WorkingPoints)

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
    EAFile_MiniIso = cms.FileInPath("RecoEgamma/ElectronIdentification/data/Run3_Winter22/effAreaElectrons_cone03_pfNeuHadronsAndPhotons_122X.txt"),
    EAFile_PFIso = cms.FileInPath("RecoEgamma/ElectronIdentification/data/Run3_Winter22/effAreaElectrons_cone03_pfNeuHadronsAndPhotons_122X.txt"),
)

isoForEleFall17V2 = isoForEle.clone(
    EAFile_MiniIso = cms.FileInPath("RecoEgamma/ElectronIdentification/data/Fall17/effAreaElectrons_cone03_pfNeuHadronsAndPhotons_94X.txt"),
    EAFile_PFIso = cms.FileInPath("RecoEgamma/ElectronIdentification/data/Fall17/effAreaElectrons_cone03_pfNeuHadronsAndPhotons_94X.txt"),
)
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

(run2_egamma_2016 & tracker_apv_vfp30_2016).toModify(
    calibratedPatElectronsNano,
    correctionFile = "EgammaAnalysis/ElectronTools/data/ScalesSmearings/Run2016_UltraLegacy_preVFP_RunFineEtaR9Gain_v3"
)

(run2_egamma_2016 & ~tracker_apv_vfp30_2016).toModify(
    calibratedPatElectronsNano,
    correctionFile = "EgammaAnalysis/ElectronTools/data/ScalesSmearings/Run2016_UltraLegacy_postVFP_RunFineEtaR9Gain_v1"
)

run2_egamma_2017.toModify(
    calibratedPatElectronsNano,
    correctionFile = "EgammaAnalysis/ElectronTools/data/ScalesSmearings/Run2017_24Feb2020_runEtaR9Gain_v2"
)

run2_egamma_2018.toModify(
    calibratedPatElectronsNano,
    correctionFile = "EgammaAnalysis/ElectronTools/data/ScalesSmearings/Run2018_29Sep2020_RunFineEtaR9Gain"
)
##############################end calibratedPatElectronsNano############################33

#####################Start slimmedElectronsWithUserData###############################3
##import from PhysicsTools/PatAlgos/python/electronsWithUserData_cfi.py
slimmedElectronsWithUserData = cms.EDProducer("PATElectronUserDataEmbedder",
    src = cms.InputTag("slimmedElectrons"),
    parentSrcs = cms.VInputTag("reducedEgamma:reducedGedGsfElectrons"),
    userFloats = cms.PSet(        
        mvaIso_Fall17V2 = cms.InputTag("electronMVAValueMapProducer:ElectronMVAEstimatorRun2Fall17IsoV2Values"),
        mvaNoIso_Fall17V2 = cms.InputTag("electronMVAValueMapProducer:ElectronMVAEstimatorRun2Fall17NoIsoV2Values"),
        mvaIso = cms.InputTag("electronMVAValueMapProducer:ElectronMVAEstimatorRun2RunIIIWinter22IsoV1Values"),
        mvaNoIso = cms.InputTag("electronMVAValueMapProducer:ElectronMVAEstimatorRun2RunIIIWinter22NoIsoV1Values"),
        mvaHZZIso = cms.InputTag("electronMVAValueMapProducer:ElectronMVAEstimatorRun2Winter22HZZV1Values"),

        miniIsoChg = cms.InputTag("isoForEle:miniIsoChg"),
        miniIsoAll = cms.InputTag("isoForEle:miniIsoAll"),
        PFIsoChg = cms.InputTag("isoForEle:PFIsoChg"),
        PFIsoAll = cms.InputTag("isoForEle:PFIsoAll"),
        PFIsoAll04 = cms.InputTag("isoForEle:PFIsoAll04"),

        miniIsoChg_Fall17V2 = cms.InputTag("isoForEleFall17V2:miniIsoChg"),
        miniIsoAll_Fall17V2 = cms.InputTag("isoForEleFall17V2:miniIsoAll"),
        PFIsoChg_Fall17V2 = cms.InputTag("isoForEleFall17V2:PFIsoChg"),
        PFIsoAll_Fall17V2 = cms.InputTag("isoForEleFall17V2:PFIsoAll"),
        PFIsoAll04_Fall17V2 = cms.InputTag("isoForEleFall17V2:PFIsoAll04"),

        ptRatio = cms.InputTag("ptRatioRelForEle:ptRatio"),
        ptRel = cms.InputTag("ptRatioRelForEle:ptRel"),
        jetNDauChargedMVASel = cms.InputTag("ptRatioRelForEle:jetNDauChargedMVASel"),
    ),
    userIntFromBools = cms.PSet(        
        mvaIso_Fall17V2_WP90 = cms.InputTag("egmGsfElectronIDs:mvaEleID-Fall17-iso-V2-wp90"),
        mvaIso_Fall17V2_WP80 = cms.InputTag("egmGsfElectronIDs:mvaEleID-Fall17-iso-V2-wp80"),
        mvaIso_Fall17V2_WPL = cms.InputTag("egmGsfElectronIDs:mvaEleID-Fall17-iso-V2-wpLoose"),
        mvaIso_WP90 = cms.InputTag("egmGsfElectronIDs:mvaEleID-RunIIIWinter22-iso-V1-wp90"),
        mvaIso_WP80 = cms.InputTag("egmGsfElectronIDs:mvaEleID-RunIIIWinter22-iso-V1-wp80"),                
        mvaNoIso_Fall17V2_WP90 = cms.InputTag("egmGsfElectronIDs:mvaEleID-Fall17-noIso-V2-wp90"),
        mvaNoIso_Fall17V2_WP80 = cms.InputTag("egmGsfElectronIDs:mvaEleID-Fall17-noIso-V2-wp80"),
        mvaNoIso_Fall17V2_WPL = cms.InputTag("egmGsfElectronIDs:mvaEleID-Fall17-noIso-V2-wpLoose"),
        mvaNoIso_WP90 = cms.InputTag("egmGsfElectronIDs:mvaEleID-RunIIIWinter22-noIso-V1-wp90"),
        mvaNoIso_WP80 = cms.InputTag("egmGsfElectronIDs:mvaEleID-RunIIIWinter22-noIso-V1-wp80"),
        mvaIso_WPHZZ = cms.InputTag("egmGsfElectronIDs:mvaEleID-Winter22-HZZ-V1"),

        cutBasedID_veto = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-RunIIIWinter22-V1-veto"),
        cutBasedID_loose = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-RunIIIWinter22-V1-loose"),
        cutBasedID_medium = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-RunIIIWinter22-V1-medium"),
        cutBasedID_tight = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-RunIIIWinter22-V1-tight"),
        cutBasedID_Fall17V2_veto = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Fall17-94X-V2-veto"),
        cutBasedID_Fall17V2_loose = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Fall17-94X-V2-loose"),
        cutBasedID_Fall17V2_medium = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Fall17-94X-V2-medium"),
        cutBasedID_Fall17V2_tight = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Fall17-94X-V2-tight"),
        cutBasedID_HEEP = cms.InputTag("egmGsfElectronIDs:heepElectronID-HEEPV70"),
    ),
    userInts = cms.PSet(
        VIDNestedWPBitmap = cms.InputTag("bitmapVIDForEle"),
        VIDNestedWPBitmap_Fall17V2 = cms.InputTag("bitmapVIDForEleFall17V2"),
        VIDNestedWPBitmapHEEP = cms.InputTag("bitmapVIDForEleHEEP"),
        seedGain = cms.InputTag("seedGainEle"),
      
    ),
    userCands = cms.PSet(
        jetForLepJetVar = cms.InputTag("ptRatioRelForEle:jetForLepJetVar") # warning: Ptr is null if no match is found
    ),
)

# no need for the Run3 IDs in Run2
run2_egamma.toModify(slimmedElectronsWithUserData.userFloats,
                     mvaIso = None,
                     mvaNoIso = None,
                     miniIsoChg = None,
                     miniIsoAll = None,
                     PFIsoChg = None,
                     PFIsoAll = None,
                     PFIsoAll04 = None).\
        toModify(slimmedElectronsWithUserData.userIntFromBools,
                 mvaIso_WPHZZ = None,
                 mvaIso_WP90 = None,
                 mvaIso_WP80 = None,
                 mvaNoIso_WP90 = None,
                 mvaNoIso_WP80 = None,
                 cutBasedID_veto = None,
                 cutBasedID_loose = None,
                 cutBasedID_medium = None,
                 cutBasedID_tight = None).\
        toModify(slimmedElectronsWithUserData.userInts,
                 VIDNestedWPBitmap = None)

run2_egamma.toModify(
    slimmedElectronsWithUserData.userFloats,
    ecalTrkEnergyErrPostCorrNew = cms.InputTag("calibratedPatElectronsNano","ecalTrkEnergyErrPostCorr"),
    ecalTrkEnergyPreCorrNew     = cms.InputTag("calibratedPatElectronsNano","ecalTrkEnergyPreCorr"),
    ecalTrkEnergyPostCorrNew    = cms.InputTag("calibratedPatElectronsNano","ecalTrkEnergyPostCorr"),
    energyScaleUpNew            = cms.InputTag("calibratedPatElectronsNano","energyScaleUp"),
    energyScaleDownNew          = cms.InputTag("calibratedPatElectronsNano","energyScaleDown"),
    energySigmaUpNew            = cms.InputTag("calibratedPatElectronsNano","energySigmaUp"),
    energySigmaDownNew          = cms.InputTag("calibratedPatElectronsNano","energySigmaDown")
)

(run2_egamma_2016).toModify(
    slimmedElectronsWithUserData.userFloats,
    mvaHZZIso = "electronMVAValueMapProducer:ElectronMVAEstimatorRun2Summer16ULIdIsoValues"
)
(run2_egamma_2017).toModify(
    slimmedElectronsWithUserData.userFloats,
    mvaHZZIso = "electronMVAValueMapProducer:ElectronMVAEstimatorRun2Summer17ULIdIsoValues"
)
(run2_egamma_2018).toModify(
    slimmedElectronsWithUserData.userFloats,
    mvaHZZIso = "electronMVAValueMapProducer:ElectronMVAEstimatorRun2Summer18ULIdIsoValues"
)
#################################################END slimmedElectrons with user data#####################

#################################################finalElectrons#####################
finalElectrons = cms.EDFilter("PATElectronRefSelector",
    src = cms.InputTag("slimmedElectronsWithUserData"),
    cut = cms.string("pt > 5 ")
)
#################################################finalElectrons#####################

################################################electronPROMPTMVA#####################
electronPROMPTMVA= cms.EDProducer("EleBaseMVAValueMapProducer",
    src = cms.InputTag("linkedObjects","electrons"),
    weightFile =  cms.FileInPath("PhysicsTools/NanoAOD/data/el_BDTG_2017.weights.xml"),
    name = cms.string("electronPROMPTMVA"),
    backend = cms.string("TMVA"),
    isClassifier = cms.bool(True),
    variables = cms.VPSet(
        cms.PSet( name = cms.string("LepGood_pt"), expr = cms.string("pt")),
        cms.PSet( name = cms.string("LepGood_eta"), expr = cms.string("eta")),
        cms.PSet( name = cms.string("LepGood_jetNDauChargedMVASel"), expr = cms.string("?userCand('jetForLepJetVar').isNonnull()?userFloat('jetNDauChargedMVASel'):0")),
        cms.PSet( name = cms.string("LepGood_miniRelIsoCharged"), expr = cms.string("userFloat('miniIsoChg_Fall17V2')/pt")),
        cms.PSet( name = cms.string("LepGood_miniRelIsoNeutral"), expr = cms.string("(userFloat('miniIsoAll_Fall17V2')-userFloat('miniIsoChg_Fall17V2'))/pt")),
        cms.PSet( name = cms.string("LepGood_jetPtRelv2"), expr = cms.string("?userCand('jetForLepJetVar').isNonnull()?userFloat('ptRel'):0")),
        cms.PSet( name = cms.string("LepGood_jetDF"), expr = cms.string("?userCand('jetForLepJetVar').isNonnull()?max(userCand('jetForLepJetVar').bDiscriminator('pfDeepFlavourJetTags:probbb')+userCand('jetForLepJetVar').bDiscriminator('pfDeepFlavourJetTags:probb')+userCand('jetForLepJetVar').bDiscriminator('pfDeepFlavourJetTags:problepb'),0.0):0.0")),
        cms.PSet( name = cms.string("LepGood_jetPtRatio"), expr = cms.string("?userCand('jetForLepJetVar').isNonnull()?min(userFloat('ptRatio'),1.5):1.0/(1.0+userFloat('PFIsoAll04_Fall17V2')/pt)")),
        cms.PSet( name = cms.string("LepGood_dxy"), expr = cms.string("log(abs(dB('PV2D')))")),
        cms.PSet( name = cms.string("LepGood_sip3d"), expr = cms.string("abs(dB('PV3D')/edB('PV3D'))")),
        cms.PSet( name = cms.string("LepGood_dz"), expr = cms.string("log(abs(dB('PVDZ')))")),
        cms.PSet( name = cms.string("LepGood_mvaFall17V2noIso"), expr = cms.string("userFloat('mvaNoIso_Fall17V2')")),
    )
)
run2_egamma_2016.toModify(
    electronPROMPTMVA,
    weightFile = "PhysicsTools/NanoAOD/data/el_BDTG_2016.weights.xml",
)
################################################electronPROMPTMVA end#####################

################################################electronTable defn #####################
electronTable = simplePATElectronFlatTableProducer.clone(
    src = cms.InputTag("linkedObjects","electrons"),
    name= cms.string("Electron"),
    doc = cms.string("slimmedElectrons after basic selection (" + finalElectrons.cut.value()+")"),
    variables = cms.PSet(CandVars,
        jetIdx = Var("?hasUserCand('jet')?userCand('jet').key():-1", "int16", doc="index of the associated jet (-1 if none)"),
        photonIdx = Var("?overlaps('photons').size()>0?overlaps('photons')[0].key():-1", "int16", doc="index of the first associated photon (-1 if none)"),
        svIdx = Var("?hasUserCand('vertex')?userCand('vertex').key():-1", "int16", doc="index of matching secondary vertex"),
        fbrem = Var("fbrem()",float,doc="Fraction of brem",precision=10),                 
        rawEnergy = Var("superCluster.rawEnergy",float,doc="raw energy of Supercluster",precision=10),
        PreshowerEnergy = Var("superCluster.preshowerEnergy",float,doc="energy deposited in preshower",precision=10),                 
        ecalEnergy = Var("ecalEnergy()",float,doc="energy after ECAL-only regression applied",precision=10),
        ecalEnergyError = Var("ecalEnergyError",float,doc="ecalEnergy error",precision=10),
        energyErr = Var("p4Error('P4_COMBINATION')",float,doc="energy error of the cluster-track combination",precision=6),
        gsfTrkpMode = Var("gsfTrack().pMode()",float,doc="GSF track pMode",precision=10),
        gsfTrkpModeErr = Var("abs(gsfTrack().qoverpModeError())*gsfTrack().pMode()*gsfTrack().pMode()",float,doc="GSF track pMode error",precision=8),
        gsfTrketaMode = Var("gsfTrack().etaMode()",float,doc="GSF track etaMode",precision=10),
        gsfTrkphiMode = Var("gsfTrack().phiMode()",float,doc="GSF track phiMode",precision=10),                 
        isEcalDriven = Var("ecalDrivenSeed",bool,doc="is ECAL driven if true"),
        isEB = Var("isEB",bool,doc="object in barrel if true derived from the seedCrystal and detID information"),                 
        dz = Var("dB('PVDZ')",float,doc="dz (with sign) wrt first PV, in cm",precision=10),
        dzErr = Var("abs(edB('PVDZ'))",float,doc="dz uncertainty, in cm",precision=6),
        dxy = Var("dB('PV2D')",float,doc="dxy (with sign) wrt first PV, in cm",precision=10),
        dxyErr = Var("edB('PV2D')",float,doc="dxy uncertainty, in cm",precision=6),
        ip3d = Var("abs(dB('PV3D'))",float,doc="3D impact parameter wrt first PV, in cm",precision=10),
        sip3d = Var("abs(dB('PV3D')/edB('PV3D'))",float,doc="3D impact parameter significance wrt first PV, in cm",precision=10),
        deltaEtaSC = Var("superCluster().eta()-eta()",float,doc="delta eta (SC,ele) with sign",precision=10),
        superclusterEta = Var("superCluster().eta()",float,doc="supercluster eta",precision=10),
        r9 = Var("full5x5_r9()",float,doc="R9 of the supercluster, calculated with full 5x5 region",precision=10),
        sieie = Var("full5x5_sigmaIetaIeta()",float,doc="sigma_IetaIeta of the supercluster, calculated with full 5x5 region",precision=10),
        eInvMinusPInv = Var("(1-eSuperClusterOverP())/ecalEnergy()",float,doc="1/E_SC - 1/p_trk",precision=10),
        scEtOverPt = Var("(superCluster().energy()/(pt*cosh(superCluster().eta())))-1",float,doc="(supercluster transverse energy)/pt-1",precision=8),

        mvaIso = Var("userFloat('mvaIso')",float,doc="MVA Iso ID score, Winter22V1"),
        mvaIso_WP80 = Var("userInt('mvaIso_WP80')",bool,doc="MVA Iso ID WP80, Winter22V1"),
        mvaIso_WP90 = Var("userInt('mvaIso_WP90')",bool,doc="MVA Iso ID WP90, Winter22V1"),
        mvaNoIso = Var("userFloat('mvaNoIso')",float,doc="MVA noIso ID score, Winter22V1"),
        mvaNoIso_WP80 = Var("userInt('mvaNoIso_WP80')",bool,doc="MVA noIso ID WP80, Winter22V1"),
        mvaNoIso_WP90 = Var("userInt('mvaNoIso_WP90')",bool,doc="MVA noIso ID WP90, Winter22V1"),                 
        mvaHZZIso = Var("userFloat('mvaHZZIso')", float,doc="HZZ MVA Iso ID score"),
        mvaIso_WPHZZ = Var("userInt('mvaIso_WPHZZ')",bool,doc="MVA Iso ID WPHZZ, Winter22V1"),

        cutBased = Var("userInt('cutBasedID_veto')+userInt('cutBasedID_loose')+userInt('cutBasedID_medium')+userInt('cutBasedID_tight')", "uint8", doc="cut-based ID RunIII Winter22 (0:fail, 1:veto, 2:loose, 3:medium, 4:tight)"),
        vidNestedWPBitmap = Var("userInt('VIDNestedWPBitmap')", int, doc=_bitmapVIDForEle_docstring),
        vidNestedWPBitmapHEEP = Var("userInt('VIDNestedWPBitmapHEEP')", int, doc=_bitmapVIDForEleHEEP_docstring),
        cutBased_HEEP = Var("userInt('cutBasedID_HEEP')",bool,doc="cut-based HEEP ID"),
        miniPFRelIso_chg = Var("userFloat('miniIsoChg')/pt",float,doc="mini PF relative isolation, charged component"),
        miniPFRelIso_all = Var("userFloat('miniIsoAll')/pt",float,doc="mini PF relative isolation, total (with scaled rho*EA PU Winter22V1 corrections)"),
        pfRelIso03_chg = Var("userFloat('PFIsoChg')/pt",float,doc="PF relative isolation dR=0.3, charged component"),
        pfRelIso03_all = Var("userFloat('PFIsoAll')/pt",float,doc="PF relative isolation dR=0.3, total (with rho*EA PU Winter22V1 corrections)"),                
        pfRelIso04_all = Var("userFloat('PFIsoAll04')/pt",float,doc="PF relative isolation dR=0.4, total (with rho*EA PU Winter22V1 corrections)", precision=10),
        jetRelIso = Var("?userCand('jetForLepJetVar').isNonnull()?(1./userFloat('ptRatio'))-1.:-1.",float,doc="Relative isolation in matched jet (1/ptRatio-1), -1 if none",precision=8),
        jetPtRelv2 = Var("?userCand('jetForLepJetVar').isNonnull()?userFloat('ptRel'):0",float,doc="Relative momentum of the lepton with respect to the closest jet after subtracting the lepton",precision=8),
        jetDF = Var("?userCand('jetForLepJetVar').isNonnull()?max(userCand('jetForLepJetVar').bDiscriminator('pfDeepFlavourJetTags:probbb')+userCand('jetForLepJetVar').bDiscriminator('pfDeepFlavourJetTags:probb')+userCand('jetForLepJetVar').bDiscriminator('pfDeepFlavourJetTags:problepb'),0.0):0.0",float,doc="value of the DEEPJET b tagging algorithm discriminator of the associated jet (0 if none)",precision=8,lazyEval=True),
        dr03TkSumPt = Var("?pt>35?dr03TkSumPt():0",float,doc="Non-PF track isolation within a delta R cone of 0.3 with electron pt > 35 GeV",precision=8),
        dr03TkSumPtHEEP = Var("?pt>35?dr03TkSumPtHEEP():0",float,doc="Non-PF track isolation within a delta R cone of 0.3 with electron pt > 35 GeV used in HEEP ID",precision=8),
        dr03EcalRecHitSumEt = Var("?pt>35?dr03EcalRecHitSumEt():0",float,doc="Non-PF Ecal isolation within a delta R cone of 0.3 with electron pt > 35 GeV",precision=8),
        dr03HcalDepth1TowerSumEt = Var("?pt>35?dr03HcalTowerSumEt(1):0",float,doc="Non-PF Hcal isolation within a delta R cone of 0.3 with electron pt > 35 GeV",precision=8),
        hoe = Var("hadronicOverEm()",float,doc="H over E",precision=8),
        tightCharge = Var("isGsfCtfScPixChargeConsistent() + isGsfScPixChargeConsistent()", "uint8", doc="Tight charge criteria (0:none, 1:isGsfScPixChargeConsistent, 2:isGsfCtfScPixChargeConsistent)"),
        convVeto = Var("passConversionVeto()",bool,doc="pass conversion veto"),
        lostHits = Var("gsfTrack.hitPattern.numberOfLostHits('MISSING_INNER_HITS')","uint8",doc="number of missing inner hits"),
        isPFcand = Var("pfCandidateRef().isNonnull()",bool,doc="electron is PF candidate"),
        seedGain = Var("userInt('seedGain')","uint8",doc="Gain of the seed crystal"),
        seediEtaOriX = Var("superCluster().seedCrysIEtaOrIx","int16",doc="iEta or iX of seed crystal. iEta is barrel-only, iX is endcap-only. iEta runs from -85 to +85, with no crystal at iEta=0. iX runs from 1 to 100."),
        seediPhiOriY = Var("superCluster().seedCrysIPhiOrIy","int16",doc="iPhi or iY of seed crystal. iPhi is barrel-only, iY is endcap-only. iPhi runs from 1 to 360. iY runs from 1 to 100."),
        jetNDauCharged = Var("?userCand('jetForLepJetVar').isNonnull()?userFloat('jetNDauChargedMVASel'):0", "uint8", doc="number of charged daughters of the closest jet"),
    ),
    externalVariables = cms.PSet(
        promptMVA = ExtVar(cms.InputTag("electronPROMPTMVA"),float, doc="Prompt MVA lepton ID score. Corresponds to the previous mvaTTH",precision=14),
        fsrPhotonIdx = ExtVar(cms.InputTag("leptonFSRphotons:eleFsrIndex"), "int16", doc="Index of the lowest-dR/ET2 among associated FSR photons"),
    ),
)

# extra variables for e/gamma custom nano
_eleVarsExtra = cms.PSet(
    r9Frac = Var("r9()",float,doc="Fractional R9 of the supercluster",precision=10),
    DeltaEtaInSC = Var("deltaEtaSuperClusterTrackAtVtx",float,doc="dEta(Inner track, supercluster)",precision=10),
    DeltaEtaInSeed = Var("deltaEtaSeedClusterTrackAtCalo",float,doc="dEta(Inner track, seedcluster)",precision=10),
    DeltaPhiInSC = Var("deltaPhiSuperClusterTrackAtVtx",float,doc="dPhi(Inner track, supercluster)",precision=10),
    DeltaPhiInSeed = Var("deltaPhiSeedClusterTrackAtCalo",float,doc="dPhi(Inner track, seedcluster)",precision=10),
    full5x5HoverE = Var("full5x5_hcalOverEcal",float,doc="full5x5 H/E",precision=10),
    eSCOverP = Var("eSuperClusterOverP",float,doc="E/P",precision=10),
    eEleOverPout = Var("eEleClusterOverPout",float,doc="EleE/Pouter",precision=10),
    e1x5 = Var("full5x5_e1x5",float,doc="energy in 1x5",precision=10),
    e2x5max = Var("full5x5_e2x5Max",float,doc="energy in 2x5",precision=10),
    e5x5 = Var("full5x5_e5x5",float,doc="energy in 5x5",precision=10),
    closestKFchi2 = Var("closestCtfTrackNormChi2",float,doc="KF track Chi2",precision=10),
    closestKFNLayers = Var("closestCtfTrackNLayers",int,doc="KF track number of layers"),
    dr03HcalTowerSumEt = Var("dr03HcalTowerSumEt",float,doc="HCal isolation",precision=10),
    GSFchi2 = Var("gsfTrack.normalizedChi2",float,doc="GSF track Chi2",precision=10),
    ecalPFClusIso = Var("ecalPFClusterIso",float,doc="ECAL PF cluster isolation",precision=10),
    hcalPFClusIso = Var("hcalPFClusterIso",float,doc="HCAL PF cluster isolation",precision=10),
    nBrem = Var("numberOfBrems",int,doc="number of brems"),
    pfPhotonIso = Var("pfIsolationVariables.sumPhotonEt",float,doc="PF photon isolation (no PU correction)",precision=10),
    pfChargedHadIso = Var("pfIsolationVariables.sumChargedHadronPt",float,doc="PF charged-hadron isolation (no PU correction)",precision=10),
    pfNeutralHadIso = Var("pfIsolationVariables.sumNeutralHadronEt",float,doc="PF neutral-hadron isolation (no PU correction)",precision=10),
    sigmaIphiIphiFull5x5 = Var("full5x5_sigmaIphiIphi",float,doc="Full5x5 sigmaIPhiIPhi",precision=10),
    etaWidth = Var("superCluster.etaWidth",float,doc="etawidth of supercluster",precision=10),
    phiWidth = Var("superCluster.phiWidth",float,doc="phiwidth of supercluster",precision=10),
    seedClusEnergy = Var("superCluster.seed.energy",float,doc="seed cluster energy",precision=10),
    hoeSingleTower = Var("hcalOverEcalBc",float,doc="Single HCAL tower based H/E",precision=10),
    hoeFull5x5 = Var("full5x5_hcalOverEcal",float,doc="Full5x5 cone-based H/E",precision=10),
    sigmaIetaIphiFull5x5 = Var("full5x5_showerShape.sigmaIetaIphi",float,doc="Full5x5 sigmaIEtaIPhi",precision=10),
    eMax = Var("full5x5_showerShape.eMax",float,doc="Emax",precision=10),
    e2nd = Var("full5x5_showerShape.e2nd",float,doc="E2nd",precision=10),
    eTop = Var("full5x5_showerShape.eTop",float,doc="Etop",precision=10),
    eBottom = Var("full5x5_showerShape.eBottom",float,doc="Ebottom",precision=10),
    eLeft = Var("full5x5_showerShape.eLeft",float,doc="Eleft",precision=10),
    eRight = Var("full5x5_showerShape.eRight",float,doc="Eright",precision=10),
    e2x5Top = Var("full5x5_showerShape.e2x5Top",float,doc="E2x5Top",precision=10),
    e2x5Bottom = Var("full5x5_showerShape.e2x5Bottom",float,doc="E2x5Bottom",precision=10),
    e2x5Left = Var("full5x5_showerShape.e2x5Left",float,doc="E2x5Left",precision=10),
    e2x5Right = Var("full5x5_showerShape.e2x5Right",float,doc="E2x5Right",precision=10),
    nSaturatedXtals = Var("nSaturatedXtals",int,doc="number of saturated crystals"),
    numberOfClusters = Var("superCluster.clusters.size",int,doc="number of clusters"),
    istrackerDriven = Var("trackerDrivenSeed",bool,doc="is tracker driven if true"),
    superclusterPhi = Var("superCluster().phi()",float,doc="supercluster phi",precision=10),
    seedClusterEta = Var("superCluster().seed().eta()",float,doc="seed cluster eta",precision=10),
    seedClusterPhi = Var("superCluster().seed().phi()",float,doc="seed cluster phi",precision=10),
    superclusterEnergy = Var("superCluster().energy()",float,doc="only PF cluster energy is corrected, no object-level regression",precision=10),
    energy = Var("energy()",float,doc="energy after final regression",precision=10),
    trackMomentumError = Var("trackMomentumError",float,doc="trackMomentum error",precision=10),
    trackMomentum = Var("trackMomentumAtVtx().R()",float,doc="trackMomentum at vertex",precision=10),
    trkLayersWithMeas = Var("gsfTrack.hitPattern.trackerLayersWithMeasurement",int,doc="trackerLayersWithMeasurement"),
    nValidPixBarrelHits = Var("gsfTrack.hitPattern.numberOfValidPixelBarrelHits",int,doc="numberOfValidPixelBarrelHits"),
    nValidPixEndcapHits = Var("gsfTrack.hitPattern.numberOfValidPixelEndcapHits",int,doc="numberOfValidPixelEndcapHits"),
    superClusterFbrem = Var("superClusterFbrem",float,doc="superClusterFbrem",precision=10),
    convVtxFitProb = Var("convVtxFitProb",float,doc="convVtxFitProb",precision=10),
    clustersSize = Var("superCluster.clustersSize",int,doc="clustersSize"),
    iEtaMod5 = Var("?superCluster.seedCrysIEtaOrIx>0?(superCluster.seedCrysIEtaOrIx-1)%5:(superCluster.seedCrysIEtaOrIx+1)%5",int,doc="iEtaMod5"),
    iEtaMod20 = Var("?abs(superCluster.seedCrysIEtaOrIx)<=25?(superCluster.seedCrysIEtaOrIx-(?superCluster.seedCrysIEtaOrIx>0?+1:-1))%20:(superCluster.seedCrysIEtaOrIx-(?superCluster.seedCrysIEtaOrIx>0?+26:-26))%20",int,doc="iEtaMod20"),
    iPhiMod2 = Var("(superCluster.seedCrysIPhiOrIy-1)%2",int,doc="iPhiMod2"),
    iPhiMod20 = Var("(superCluster.seedCrysIPhiOrIy-1)%20",int,doc="iPhiMod20"),
)

(run2_egamma).toModify(
        # energy scale/smearing: only for Run2
        electronTable.variables,
        pt = Var("pt*userFloat('ecalTrkEnergyPostCorrNew')/userFloat('ecalTrkEnergyPreCorrNew')", float, precision=-1, doc="p_{T}"),
        energyErr = Var("userFloat('ecalTrkEnergyErrPostCorrNew')", float, precision=6, doc="energy error of the cluster-track combination"),
        ptPreCorr = Var("pt", float, doc="pt of the electron before energy corrections"),
        scEtOverPt = Var("(superCluster().energy()/(pt*userFloat('ecalTrkEnergyPostCorrNew')/userFloat('ecalTrkEnergyPreCorrNew')*cosh(superCluster().eta())))-1",float,doc="(supercluster transverse energy)/pt-1",precision=8),
        dEscaleUp=Var("userFloat('ecalTrkEnergyPostCorrNew')-userFloat('energyScaleUpNew')", float,  doc="ecal energy scale shifted 1 sigma up(adding gain/stat/syst in quadrature)", precision=8),
        dEscaleDown=Var("userFloat('ecalTrkEnergyPostCorrNew')-userFloat('energyScaleDownNew')", float,  doc="ecal energy scale shifted 1 sigma down (adding gain/stat/syst in quadrature)", precision=8),
        dEsigmaUp=Var("userFloat('ecalTrkEnergyPostCorrNew')-userFloat('energySigmaUpNew')", float, doc="ecal energy smearing value shifted 1 sigma up", precision=8),
        dEsigmaDown=Var("userFloat('ecalTrkEnergyPostCorrNew')-userFloat('energySigmaDownNew')", float,  doc="ecal energy smearing value shifted 1 sigma up", precision=8),
        # Fall17V2 IDs and isolations are only for Run2. The names of these IDs and isolations are same as in Run3. 
        mvaIso = Var("userFloat('mvaIso_Fall17V2')",float,doc="MVA Iso ID score, Fall17V2"),
        mvaIso_WP80 = Var("userInt('mvaIso_Fall17V2_WP80')",bool,doc="MVA Iso ID WP80, Fall17V2"),
        mvaIso_WP90 = Var("userInt('mvaIso_Fall17V2_WP90')",bool,doc="MVA Iso ID WP90, Fall17V2"),
        mvaIso_WPL = Var("userInt('mvaIso_Fall17V2_WPL')",bool,doc="MVA Iso ID loose WP, Fall17V2"),
        mvaNoIso = Var("userFloat('mvaNoIso_Fall17V2')",float,doc="MVA noIso ID score, Fall17V2"),
        mvaNoIso_WP80 = Var("userInt('mvaNoIso_Fall17V2_WP80')",bool,doc="MVA noIso ID WP80, Fall17V2"),
        mvaIso_WPHZZ = None,
        mvaNoIso_WP90 = Var("userInt('mvaNoIso_Fall17V2_WP90')",bool,doc="MVA noIso ID WP90, Fall17V2"),
        mvaNoIso_WPL = Var("userInt('mvaNoIso_Fall17V2_WPL')",bool,doc="MVA noIso ID loose WP, Fall17V2"),
        cutBased = Var("userInt('cutBasedID_Fall17V2_veto')+userInt('cutBasedID_Fall17V2_loose')+userInt('cutBasedID_Fall17V2_medium')+userInt('cutBasedID_Fall17V2_tight')", "uint8", doc="cut-based ID Fall17V2 (0:fail, 1:veto, 2:loose, 3:medium, 4:tight)"),
        vidNestedWPBitmap = Var("userInt('VIDNestedWPBitmap_Fall17V2')", int, doc=_bitmapVIDForEleFall17V2_docstring),
        miniPFRelIso_chg = Var("userFloat('miniIsoChg_Fall17V2')/pt",float,doc="mini PF relative isolation, charged component in Run2"),
        miniPFRelIso_all = Var("userFloat('miniIsoAll_Fall17V2')/pt",float,doc="mini PF relative isolation, total (with scaled rho*EA Fall17V2 PU corrections) in Run2"),
        pfRelIso03_chg = Var("userFloat('PFIsoChg_Fall17V2')/pt",float,doc="PF relative isolation dR=0.3 with 94 EffArea, charged component in Run2"),
        pfRelIso03_all = Var("userFloat('PFIsoAll_Fall17V2')/pt",float,doc="PF relative isolation dR=0.3 with 94 EffArea, total (with rho*EA Fall17V2 PU corrections) in Run2"),
        pfRelIso04_all = Var("userFloat('PFIsoAll04_Fall17V2')/pt",float,doc="PF relative isolation dR=0.4, total (with rho*EA PU Fall17V2 PU corrections)", precision=10),
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

electronTask = cms.Task(bitmapVIDForEle,bitmapVIDForEleFall17V2,bitmapVIDForEleHEEP,isoForEle,isoForEleFall17V2,ptRatioRelForEle,seedGainEle,calibratedPatElectronsNano,slimmedElectronsWithUserData,finalElectrons)
electronTablesTask = cms.Task(electronPROMPTMVA, electronTable)
electronMCTask = cms.Task(tautaggerForMatching, matchingElecPhoton, electronsMCMatchForTable, electronsMCMatchForTableAlt, electronMCTable)

_electronTask_Run2 = electronTask.copy()
_electronTask_Run2.remove(bitmapVIDForEle)
_electronTask_Run2.remove(isoForEle)
_electronTask_Run2.add(calibratedPatElectronsNano)
run2_egamma.toReplaceWith(electronTask, _electronTask_Run2)

# Revert back to AK4 CHS jets for Run2 inputs
run2_nanoAOD_ANY.toModify(
    ptRatioRelForEle,srcJet="updatedJets")
