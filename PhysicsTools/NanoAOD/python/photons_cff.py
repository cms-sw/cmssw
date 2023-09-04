import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.nano_eras_cff import *
from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.simpleCandidateFlatTableProducer_cfi import simpleCandidateFlatTableProducer
from math import ceil,log


photon_id_modules_WorkingPoints_nanoAOD = cms.PSet(
    modules = cms.vstring(
        'RecoEgamma.PhotonIdentification.Identification.cutBasedPhotonID_RunIIIWinter22_122X_V1_cff',
        'RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Winter22_122X_V1_cff',
        # Fall17: need to include the modules too to make sure they are run
        'RecoEgamma.PhotonIdentification.Identification.cutBasedPhotonID_Fall17_94X_V2_cff',
        'RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Fall17_94X_V2_cff',
   ),
   WorkingPoints = cms.vstring(
     "egmPhotonIDs:cutBasedPhotonID-RunIIIWinter22-122X-V1-loose",
     "egmPhotonIDs:cutBasedPhotonID-RunIIIWinter22-122X-V1-medium",
     "egmPhotonIDs:cutBasedPhotonID-RunIIIWinter22-122X-V1-tight",
   )
)

photon_id_modules_WorkingPoints_nanoAOD_Run2 = cms.PSet(
    modules = cms.vstring(
        'RecoEgamma.PhotonIdentification.Identification.cutBasedPhotonID_Fall17_94X_V2_cff',
        'RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Fall17_94X_V2_cff',
   ),
    WorkingPoints = cms.vstring(
      "egmPhotonIDs:cutBasedPhotonID-Fall17-94X-V2-loose",
      "egmPhotonIDs:cutBasedPhotonID-Fall17-94X-V2-medium",
      "egmPhotonIDs:cutBasedPhotonID-Fall17-94X-V2-tight",
    )

)

# make Fall17 the default one in Run2
run2_egamma.toModify(photon_id_modules_WorkingPoints_nanoAOD,
                     modules=photon_id_modules_WorkingPoints_nanoAOD_Run2.modules).\
        toModify(photon_id_modules_WorkingPoints_nanoAOD,
                 WorkingPoints=photon_id_modules_WorkingPoints_nanoAOD_Run2.WorkingPoints)

def make_bitmapVID_docstring(id_modules_working_points_pset):
    pset = id_modules_working_points_pset

    for modname in pset.modules:
        ids = __import__(modname, globals(), locals(), ['idName','cutFlow'])
        for name in dir(ids):
            _id = getattr(ids,name)
            if hasattr(_id,'idName') and hasattr(_id,'cutFlow'):
                if (len(pset.WorkingPoints)>0 and _id.idName == pset.WorkingPoints[0].split(':')[-1]):
                    cut_names = ','.join([cut.cutName.value() for cut in _id.cutFlow])
                    n_bits_per_cut = int(ceil(log(len(pset.WorkingPoints)+1,2)))
                    return 'VID compressed bitmap (%s), %d bits per cut'%(cut_names, n_bits_per_cut)
    raise ValueError("Something is wrong in the photon ID modules parameter set!")


bitmapVIDForPho = cms.EDProducer("PhoVIDNestedWPBitmapProducer",
    src = cms.InputTag("slimmedPhotons"),
    srcForID = cms.InputTag("reducedEgamma","reducedGedPhotons"),
    WorkingPoints = photon_id_modules_WorkingPoints_nanoAOD.WorkingPoints,
)
_bitmapVIDForPho_docstring = make_bitmapVID_docstring(photon_id_modules_WorkingPoints_nanoAOD)

bitmapVIDForPhoRun2 = bitmapVIDForPho.clone(
    WorkingPoints = photon_id_modules_WorkingPoints_nanoAOD_Run2.WorkingPoints,
)
_bitmapVIDForPhoRun2_docstring = make_bitmapVID_docstring(photon_id_modules_WorkingPoints_nanoAOD_Run2)

isoForPho = cms.EDProducer("PhoIsoValueMapProducer",
    src = cms.InputTag("slimmedPhotons"),
    relative = cms.bool(False),
    doQuadratic = cms.bool(True),
    rho_PFIso = cms.InputTag("fixedGridRhoFastjetAll"),
    QuadraticEAFile_PFIso_Chg  = cms.FileInPath("RecoEgamma/PhotonIdentification/data/RunIII_Winter22/effectiveArea_ChgHadronIso_95percentBased.txt"),
    QuadraticEAFile_PFIso_ECal = cms.FileInPath("RecoEgamma/PhotonIdentification/data/RunIII_Winter22/effectiveArea_ECalClusterIso_95percentBased.txt"),
    QuadraticEAFile_PFIso_HCal = cms.FileInPath("RecoEgamma/PhotonIdentification/data/RunIII_Winter22/effectiveArea_HCalClusterIso_95percentBased.txt"),
)

hOverEForPho = cms.EDProducer("PhoHoverEValueMapProducer",
    src = cms.InputTag("slimmedPhotons"),
    relative = cms.bool(False),
    rho = cms.InputTag("fixedGridRhoFastjetAll"),
    QuadraticEAFile_HoverE = cms.FileInPath("RecoEgamma/PhotonIdentification/data/RunIII_Winter22/effectiveArea_coneBasedHoverE_95percentBased.txt"),
)

isoForPhoFall17V2 = isoForPho.clone(
    doQuadratic = cms.bool(False),
    EAFile_PFIso_Chg = cms.FileInPath("RecoEgamma/PhotonIdentification/data/Fall17/effAreaPhotons_cone03_pfChargedHadrons_90percentBased_V2.txt"),
    EAFile_PFIso_Neu = cms.FileInPath("RecoEgamma/PhotonIdentification/data/Fall17/effAreaPhotons_cone03_pfNeutralHadrons_90percentBased_V2.txt"),
    EAFile_PFIso_Pho = cms.FileInPath("RecoEgamma/PhotonIdentification/data/Fall17/effAreaPhotons_cone03_pfPhotons_90percentBased_V2.txt"),
)


seedGainPho = cms.EDProducer("PhotonSeedGainProducer", src = cms.InputTag("slimmedPhotons"))

import RecoEgamma.EgammaTools.calibratedEgammas_cff

calibratedPatPhotonsNano = RecoEgamma.EgammaTools.calibratedEgammas_cff.calibratedPatPhotons.clone(
    produceCalibratedObjs = False,
    correctionFile = cms.string("EgammaAnalysis/ElectronTools/data/ScalesSmearings/Run2016_UltraLegacy_preVFP_RunFineEtaR9Gain"),
)

(run2_egamma_2016 & tracker_apv_vfp30_2016).toModify(
    calibratedPatPhotonsNano,
    correctionFile = cms.string("EgammaAnalysis/ElectronTools/data/ScalesSmearings/Run2016_UltraLegacy_preVFP_RunFineEtaR9Gain")
)

(run2_egamma_2016 & ~tracker_apv_vfp30_2016).toModify(
    calibratedPatPhotonsNano,
    correctionFile = cms.string("EgammaAnalysis/ElectronTools/data/ScalesSmearings/Run2016_UltraLegacy_postVFP_RunFineEtaR9Gain"),
)

run2_egamma_2017.toModify(
    calibratedPatPhotonsNano,
    correctionFile = cms.string("EgammaAnalysis/ElectronTools/data/ScalesSmearings/Run2017_24Feb2020_runEtaR9Gain_v2")
)

run2_egamma_2018.toModify(
    calibratedPatPhotonsNano,
    correctionFile = cms.string("EgammaAnalysis/ElectronTools/data/ScalesSmearings/Run2018_29Sep2020_RunFineEtaR9Gain")
)

slimmedPhotonsWithUserData = cms.EDProducer("PATPhotonUserDataEmbedder",
    src = cms.InputTag("slimmedPhotons"),
    parentSrcs = cms.VInputTag("reducedEgamma:reducedGedPhotons"),
    userFloats = cms.PSet(
        mvaID = cms.InputTag("photonMVAValueMapProducer:PhotonMVAEstimatorRunIIIWinter22v1Values"),
        PFIsoChgQuadratic = cms.InputTag("isoForPho:PFIsoChgQuadratic"),
        PFIsoAllQuadratic = cms.InputTag("isoForPho:PFIsoAllQuadratic"),
        HoverEQuadratic = cms.InputTag("hOverEForPho:HoEForPhoEACorr"),
        mvaID_Fall17V2 = cms.InputTag("photonMVAValueMapProducer:PhotonMVAEstimatorRunIIFall17v2Values"),
        PFIsoChgFall17V2 = cms.InputTag("isoForPhoFall17V2:PFIsoChg"),
        PFIsoAllFall17V2 = cms.InputTag("isoForPhoFall17V2:PFIsoAll"),
    ),
    userIntFromBools = cms.PSet(
        cutBasedID_loose  = cms.InputTag("egmPhotonIDs:cutBasedPhotonID-RunIIIWinter22-122X-V1-loose"),
        cutBasedID_medium = cms.InputTag("egmPhotonIDs:cutBasedPhotonID-RunIIIWinter22-122X-V1-medium"),
        cutBasedID_tight  = cms.InputTag("egmPhotonIDs:cutBasedPhotonID-RunIIIWinter22-122X-V1-tight"),
        cutBasedID_Fall17V2_loose  = cms.InputTag("egmPhotonIDs:cutBasedPhotonID-Fall17-94X-V2-loose"),
        cutBasedID_Fall17V2_medium = cms.InputTag("egmPhotonIDs:cutBasedPhotonID-Fall17-94X-V2-medium"),
        cutBasedID_Fall17V2_tight  = cms.InputTag("egmPhotonIDs:cutBasedPhotonID-Fall17-94X-V2-tight"),
        mvaID_WP90 = cms.InputTag("egmPhotonIDs:mvaPhoID-RunIIIWinter22-v1-wp90"),
        mvaID_WP80 = cms.InputTag("egmPhotonIDs:mvaPhoID-RunIIIWinter22-v1-wp80"),
        mvaID_Fall17V2_WP90 = cms.InputTag("egmPhotonIDs:mvaPhoID-RunIIFall17-v2-wp90"),
        mvaID_Fall17V2_WP80 = cms.InputTag("egmPhotonIDs:mvaPhoID-RunIIFall17-v2-wp80"),
    ),
    userInts = cms.PSet(
        VIDNestedWPBitmap = cms.InputTag("bitmapVIDForPho"),
        VIDNestedWPBitmapFall17V2 = cms.InputTag("bitmapVIDForPhoRun2"),
        seedGain = cms.InputTag("seedGainPho"),
       
    )
)

# no need for the Run3 IDs in Run2
run2_egamma.toModify(slimmedPhotonsWithUserData.userFloats,
                     mvaID = None,
                     PFIsoChgQuadratic = None,
                     PFIsoAllQuadratic = None,
                     HoverEQuadratic = None).\
                toModify(slimmedPhotonsWithUserData.userIntFromBools,
                          cutBasedID_loose = None,
                          cutBasedID_medium = None,
                          cutBasedID_tight = None,
                          mvaID_WP90 = None,
                          mvaID_WP80 = None).\
                toModify(slimmedPhotonsWithUserData.userInts,
                         VIDNestedWPBitmap = None)

run2_egamma.toModify(
    slimmedPhotonsWithUserData.userFloats,
    ecalEnergyErrPostCorrNew = cms.InputTag("calibratedPatPhotonsNano","ecalEnergyErrPostCorr"),
    ecalEnergyPreCorrNew     = cms.InputTag("calibratedPatPhotonsNano","ecalEnergyPreCorr"),
    ecalEnergyPostCorrNew    = cms.InputTag("calibratedPatPhotonsNano","ecalEnergyPostCorr"),
    energyScaleUpNew            = cms.InputTag("calibratedPatPhotonsNano","energyScaleUp"),
    energyScaleDownNew          = cms.InputTag("calibratedPatPhotonsNano","energyScaleDown"),
    energySigmaUpNew            = cms.InputTag("calibratedPatPhotonsNano","energySigmaUp"),
    energySigmaDownNew          = cms.InputTag("calibratedPatPhotonsNano","energySigmaDown"),
)


finalPhotons = cms.EDFilter("PATPhotonRefSelector",
    src = cms.InputTag("slimmedPhotonsWithUserData"),
    cut = cms.string("pt > 5 ")
)

photonTable = simpleCandidateFlatTableProducer.clone(
    src = cms.InputTag("linkedObjects","photons"),
    name= cms.string("Photon"),
    doc = cms.string("slimmedPhotons after basic selection (" + finalPhotons.cut.value()+")"),
    variables = cms.PSet(P3Vars,
        jetIdx = Var("?hasUserCand('jet')?userCand('jet').key():-1", "int16", doc="index of the associated jet (-1 if none)"),
        electronIdx = Var("?hasUserCand('electron')?userCand('electron').key():-1", "int16", doc="index of the associated electron (-1 if none)"),
        energyErr = Var("getCorrectedEnergyError('regression2')",float,doc="energy error of the cluster from regression",precision=6),
        energyRaw = Var("superCluster().rawEnergy()",float,doc="raw energy of photon supercluster", precision=10),
        r9 = Var("full5x5_r9()",float,doc="R9 of the supercluster, calculated with full 5x5 region",precision=8),
        sieie = Var("full5x5_sigmaIetaIeta()",float,doc="sigma_IetaIeta of the supercluster, calculated with full 5x5 region",precision=8),
        sipip = Var("showerShapeVariables().sigmaIphiIphi", float, doc="sigmaIphiIphi of the supercluster", precision=8),
        sieip = Var("full5x5_showerShapeVariables().sigmaIetaIphi",float,doc="sigma_IetaIphi of the supercluster, calculated with full 5x5 region",precision=8),
        s4 = Var("full5x5_showerShapeVariables().e2x2/full5x5_showerShapeVariables().e5x5",float,doc="e2x2/e5x5 of the supercluster, calculated with full 5x5 region",precision=8),
        etaWidth = Var("superCluster().etaWidth()",float,doc="Width of the photon supercluster in eta", precision=8),
        phiWidth = Var("superCluster().phiWidth()",float,doc="Width of the photon supercluster in phi", precision=8),
        cutBased = Var(
            "userInt('cutBasedID_loose')+userInt('cutBasedID_medium')+userInt('cutBasedID_tight')",
            "uint8",
            doc="cut-based ID bitmap, RunIIIWinter22V1, (0:fail, 1:loose, 2:medium, 3:tight)",
        ),
        vidNestedWPBitmap = Var(
            "userInt('VIDNestedWPBitmap')",
            int,
            doc="RunIIIWinter22V1 " + _bitmapVIDForPho_docstring
        ),
        electronVeto = Var("passElectronVeto()",bool,doc="pass electron veto"),
        pixelSeed = Var("hasPixelSeed()",bool,doc="has pixel seed"),
        hasConversionTracks = Var("hasConversionTracks()",bool,doc="Variable specifying if photon has associated conversion tracks (one-legged or two-legged)"),
        mvaID = Var("userFloat('mvaID')",float,doc="MVA ID score, Winter22V1",precision=10),
        mvaID_WP90 = Var("userInt('mvaID_WP90')",bool,doc="MVA ID WP90, Winter22V1"),
        mvaID_WP80 = Var("userInt('mvaID_WP80')",bool,doc="MVA ID WP80, Winter22V1"),
        trkSumPtHollowConeDR03 = Var("trkSumPtHollowConeDR03()",float,doc="Sum of track pT in a hollow cone of outer radius, inner radius", precision=8),
        trkSumPtSolidConeDR04 = Var("trkSumPtSolidConeDR04()",float,doc="Sum of track pT in a cone of dR=0.4", precision=8),
        ecalPFClusterIso = Var("ecalPFClusterIso()",float,doc="sum pt of ecal clusters, vetoing clusters part of photon", precision=8),
        hcalPFClusterIso = Var("hcalPFClusterIso()",float,doc="sum pt of hcal clusters, vetoing clusters part of photon", precision=8),
        pfPhoIso03 = Var("photonIso()",float,doc="PF absolute isolation dR=0.3, photon component (uncorrected)"),
        pfChargedIso = Var("chargedHadronIso()",float,doc="PF absolute isolation dR=0.3, charged component with dxy,dz match to PV", precision=8),
        pfChargedIsoPFPV = Var("chargedHadronPFPVIso()",float,doc="PF absolute isolation dR=0.3, charged component (PF PV only)"),
        pfChargedIsoWorstVtx = Var("chargedHadronWorstVtxIso()",float,doc="PF absolute isolation dR=0.3, charged component (Vertex with largest isolation)"),
        pfRelIso03_chg_quadratic = Var("userFloat('PFIsoChgQuadratic')/pt",float,doc="PF relative isolation dR=0.3, charged hadron component (with quadraticEA*rho*rho + linearEA*rho Winter22V1 corrections)"),
        pfRelIso03_all_quadratic = Var("userFloat('PFIsoAllQuadratic')/pt",float,doc="PF relative isolation dR=0.3, total (with quadraticEA*rho*rho + linearEA*rho Winter22V1 corrections)"),
        hoe = Var("hadronicOverEm()",float,doc="H over E",precision=8),
        hoe_PUcorr = Var("userFloat('HoverEQuadratic')",float,doc="PU corrected H/E (cone-based with quadraticEA*rho*rho + linearEA*rho Winter22V1 corrections)",precision=8),
        isScEtaEB = Var("abs(superCluster().eta()) < 1.4442",bool,doc="is supercluster eta within barrel acceptance"),
        isScEtaEE = Var("abs(superCluster().eta()) > 1.566 && abs(superCluster().eta()) < 2.5",bool,doc="is supercluster eta within endcap acceptance"),
        seedGain = Var("userInt('seedGain')","uint8",doc="Gain of the seed crystal"),
        seediEtaOriX = Var("superCluster().seedCrysIEtaOrIx","int8",doc="iEta or iX of seed crystal. iEta is barrel-only, iX is endcap-only. iEta runs from -85 to +85, with no crystal at iEta=0. iX runs from 1 to 100."),
        seediPhiOriY = Var("superCluster().seedCrysIPhiOrIy",int,doc="iPhi or iY of seed crystal. iPhi is barrel-only, iY is endcap-only. iPhi runs from 1 to 360. iY runs from 1 to 100."),
        # position of photon is best approximated by position of seed cluster, not the SC centroid
        x_calo = Var("superCluster().seed().position().x()",float,doc="photon supercluster position on calorimeter, x coordinate (cm)",precision=10),
        y_calo = Var("superCluster().seed().position().y()",float,doc="photon supercluster position on calorimeter, y coordinate (cm)",precision=10),
        z_calo = Var("superCluster().seed().position().z()",float,doc="photon supercluster position on calorimeter, z coordinate (cm)",precision=10),
        # ES variables
        esEffSigmaRR = Var("full5x5_showerShapeVariables().effSigmaRR()", float, doc="preshower sigmaRR"),
        esEnergyOverRawE = Var("superCluster().preshowerEnergy()/superCluster().rawEnergy()", float, doc="ratio of preshower energy to raw supercluster energy"),
        haloTaggerMVAVal = Var("haloTaggerMVAVal()",float,doc="Value of MVA based BDT based  beam halo tagger in the Ecal endcap (valid for pT > 200 GeV)",precision=8),
    )
)


#these eras need to make the energy correction, hence the "New". Also save only Fall17V2 IDS in Run2, No Run3 Winter22V1 and quadratic iso in Run2
run2_egamma.toModify(
    photonTable.variables,
    pt = Var("pt*userFloat('ecalEnergyPostCorrNew')/userFloat('ecalEnergyPreCorrNew')", float, precision=-1, doc="p_{T}"),
    energyErr = Var("userFloat('ecalEnergyErrPostCorrNew')",float,doc="energy error of the cluster from regression",precision=6),
    eCorr = Var("userFloat('ecalEnergyPostCorrNew')/userFloat('ecalEnergyPreCorrNew')",float,doc="ratio of the calibrated energy/miniaod energy"),
    hoe = Var("hadTowOverEm()",float,doc="H over E (Run2)",precision=8),
    cutBased = Var(
            "userInt('cutBasedID_Fall17V2_loose')+userInt('cutBasedID_Fall17V2_medium')+userInt('cutBasedID_Fall17V2_tight')",
            "uint8",
            doc="cut-based ID bitmap, Fall17V2, (0:fail, 1:loose, 2:medium, 3:tight)",
        ),
    vidNestedWPBitmap = Var(
            "userInt('VIDNestedWPBitmapFall17V2')",
            int,
            doc="Fall17V2 " + _bitmapVIDForPhoRun2_docstring
        ),
    mvaID = Var("userFloat('mvaID_Fall17V2')",float,doc="MVA ID score, Fall17V2",precision=10),
    mvaID_WP90 = Var("userInt('mvaID_Fall17V2_WP90')",bool,doc="MVA ID WP90, Fall17V2"),
    mvaID_WP80 = Var("userInt('mvaID_Fall17V2_WP80')",bool,doc="MVA ID WP80, Fall17V2"),
    pfRelIso03_chg = Var("userFloat('PFIsoChgFall17V2')/pt",float,doc="PF relative isolation dR=0.3, charged component (with Fall17V2rho*EA PUcorrections)"),
    pfRelIso03_all = Var("userFloat('PFIsoAllFall17V2')/pt",float,doc="PF relative isolation dR=0.3, total (with Fall17V2 rho*EA PU corrections)"),
    pfRelIso03_chg_quadratic=None,
    pfRelIso03_all_quadratic=None,
    hoe_PUcorr=None

)

photonsMCMatchForTable = cms.EDProducer("MCMatcher",  # cut on deltaR, deltaPt/Pt; pick best by deltaR
    src         = photonTable.src,                 # final reco collection
    matched     = cms.InputTag("finalGenParticles"), # final mc-truth particle collection
    mcPdgId     = cms.vint32(11,22),                 # one or more PDG ID (11 = el, 22 = pho); absolute values (see below)
    checkCharge = cms.bool(False),              # True = require RECO and MC objects to have the same charge
    mcStatus    = cms.vint32(1),                # PYTHIA status code (1 = stable, 2 = shower, 3 = hard scattering)
    maxDeltaR   = cms.double(0.3),              # Minimum deltaR for the match
    maxDPtRel   = cms.double(0.5),              # Minimum deltaPt/Pt for the match
    resolveAmbiguities    = cms.bool(True),     # Forbid two RECO objects to match to the same GEN object
    resolveByMatchQuality = cms.bool(True),    # False = just match input in order; True = pick lowest deltaR pair first
)

photonMCTable = cms.EDProducer("CandMCMatchTableProducer",
    src     = photonTable.src,
    mcMap   = cms.InputTag("photonsMCMatchForTable"),
    objName = photonTable.name,
    objType = photonTable.name, #cms.string("Photon"),
    branchName = cms.string("genPart"),
    docString = cms.string("MC matching to status==1 photons or electrons"),
)

#adding 4 most imp scale & smearing variables to table
run2_egamma.toModify(
    photonTable.variables,
    dEscaleUp=Var("userFloat('ecalEnergyPostCorrNew') - userFloat('energyScaleUpNew')", float, doc="ecal energy scale shifted 1 sigma up (adding gain/stat/syst in quadrature)", precision=8),
    dEscaleDown=Var("userFloat('ecalEnergyPostCorrNew') - userFloat('energyScaleDownNew')", float, doc="ecal energy scale shifted 1 sigma down (adding gain/stat/syst in quadrature)", precision=8),
    dEsigmaUp=Var("userFloat('ecalEnergyPostCorrNew') - userFloat('energySigmaUpNew')", float, doc="ecal energy smearing value shifted 1 sigma up", precision=8),
    dEsigmaDown=Var("userFloat('ecalEnergyPostCorrNew') - userFloat('energySigmaDownNew')", float, doc="ecal energy smearing value shifted 1 sigma up", precision=8),
)


photonTask = cms.Task(bitmapVIDForPho, bitmapVIDForPhoRun2, isoForPho, hOverEForPho, isoForPhoFall17V2, seedGainPho, slimmedPhotonsWithUserData, finalPhotons)

photonTablesTask = cms.Task(photonTable)
photonMCTask = cms.Task(photonsMCMatchForTable, photonMCTable)

_photonTask_Run2 = photonTask.copy()
_photonTask_Run2.remove(bitmapVIDForPho)
_photonTask_Run2.remove(isoForPho)
_photonTask_Run2.remove(hOverEForPho)
_photonTask_Run2.add(calibratedPatPhotonsNano)
run2_egamma.toReplaceWith(photonTask, _photonTask_Run2)
