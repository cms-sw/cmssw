import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Modifier_run2_miniAOD_80XLegacy_cff import run2_miniAOD_80XLegacy
from Configuration.Eras.Modifier_run2_nanoAOD_92X_cff import run2_nanoAOD_92X
from PhysicsTools.NanoAOD.common_cff import *
from math import ceil,log

from PhysicsTools.SelectorUtils.tools.vid_id_tools import setupVIDSelection
from RecoEgamma.ElectronIdentification.egmGsfElectronIDs_cff import *
electronMVAValueMapProducer.srcMiniAOD = cms.InputTag("slimmedElectrons")
egmGsfElectronIDs.physicsObjectIDs = cms.VPSet()
egmGsfElectronIDs.physicsObjectSrc = cms.InputTag('slimmedElectrons')
_electron_id_vid_modules=[
'RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Summer16_80X_V1_cff',
'RecoEgamma.ElectronIdentification.Identification.cutBasedElectronHLTPreselecition_Summer16_V1_cff',
#'RecoEgamma.ElectronIdentification.Identification.heepElectronID_HEEPV70_cff', # add heepIDVarValueMaps to sequence when uncomment
'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring16_GeneralPurpose_V1_cff',
'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring16_HZZ_V1_cff',
]
_bitmapVIDForEle_WorkingPoints = cms.vstring(
    "egmGsfElectronIDs:cutBasedElectronID-Summer16-80X-V1-loose",
    "egmGsfElectronIDs:cutBasedElectronID-Summer16-80X-V1-medium",
    "egmGsfElectronIDs:cutBasedElectronID-Summer16-80X-V1-tight",
)
_bitmapVIDForEle_docstring = ''
for modname in _electron_id_vid_modules: 
    ids= __import__(modname, globals(), locals(), ['idName','cutFlow'])
    for name in dir(ids):
        _id = getattr(ids,name)
        if hasattr(_id,'idName') and hasattr(_id,'cutFlow'):
            setupVIDSelection(egmGsfElectronIDs,_id)
            if (len(_bitmapVIDForEle_WorkingPoints)>0 and _id.idName==_bitmapVIDForEle_WorkingPoints[0].split(':')[-1]):
                _bitmapVIDForEle_docstring = 'VID compressed bitmap (%s), %d bits per cut'%(','.join([cut.cutName.value() for cut in _id.cutFlow]),int(ceil(log(len(_bitmapVIDForEle_WorkingPoints)+1,2))))
from RecoEgamma.ElectronIdentification.heepIdVarValueMapProducer_cfi import *

bitmapVIDForEle = cms.EDProducer("EleVIDNestedWPBitmapProducer",
    src = cms.InputTag("slimmedElectrons"),
    WorkingPoints = _bitmapVIDForEle_WorkingPoints,
)

isoForEle = cms.EDProducer("EleIsoValueMapProducer",
    src = cms.InputTag("slimmedElectrons"),
    relative = cms.bool(False),
    rho_MiniIso = cms.InputTag("fixedGridRhoFastjetCentralNeutral"),
    rho_PFIso = cms.InputTag("fixedGridRhoFastjetAll"),
    EAFile_MiniIso = cms.FileInPath("RecoEgamma/ElectronIdentification/data/Spring15/effAreaElectrons_cone03_pfNeuHadronsAndPhotons_25ns.txt"),
    EAFile_PFIso = cms.FileInPath("RecoEgamma/ElectronIdentification/data/Summer16/effAreaElectrons_cone03_pfNeuHadronsAndPhotons_80X.txt"),
)

ptRatioRelForEle = cms.EDProducer("ElectronJetVarProducer",
    srcJet = cms.InputTag("slimmedJets"),
    srcLep = cms.InputTag("slimmedElectrons"),
    srcVtx = cms.InputTag("offlineSlimmedPrimaryVertices"),
)

from EgammaAnalysis.ElectronTools.calibratedElectronsRun2_cfi import calibratedPatElectrons
calibratedPatElectrons.correctionFile = cms.string("PhysicsTools/NanoAOD/data/80X_ichepV1_2016_ele") # hack, should go somewhere in EgammaAnalysis

energyCorrForEle =  cms.EDProducer("ElectronEnergyVarProducer",
    srcRaw = cms.InputTag("slimmedElectrons"),
    srcCorr = cms.InputTag("calibratedPatElectrons"),
)


slimmedElectronsWithUserData = cms.EDProducer("PATElectronUserDataEmbedder",
    src = cms.InputTag("slimmedElectrons"),
    userFloats = cms.PSet(
        mvaSpring16GP = cms.InputTag("electronMVAValueMapProducer:ElectronMVAEstimatorRun2Spring16GeneralPurposeV1Values"),
        mvaSpring16HZZ = cms.InputTag("electronMVAValueMapProducer:ElectronMVAEstimatorRun2Spring16HZZV1Values"),
        miniIsoChg = cms.InputTag("isoForEle:miniIsoChg"),
        miniIsoAll = cms.InputTag("isoForEle:miniIsoAll"),
        PFIsoChg = cms.InputTag("isoForEle:PFIsoChg"),
        PFIsoAll = cms.InputTag("isoForEle:PFIsoAll"),
        ptRatio = cms.InputTag("ptRatioRelForEle:ptRatio"),
        ptRel = cms.InputTag("ptRatioRelForEle:ptRel"),
        jetNDauChargedMVASel = cms.InputTag("ptRatioRelForEle:jetNDauChargedMVASel"),
        eCorr = cms.InputTag("energyCorrForEle:eCorr"),
    ),
    userIntFromBools = cms.PSet(
        mvaSpring16GP_WP90 = cms.InputTag("egmGsfElectronIDs:mvaEleID-Spring16-GeneralPurpose-V1-wp90"),
        mvaSpring16GP_WP80 = cms.InputTag("egmGsfElectronIDs:mvaEleID-Spring16-GeneralPurpose-V1-wp80"),
        mvaSpring16HZZ_WPL = cms.InputTag("egmGsfElectronIDs:mvaEleID-Spring16-HZZ-V1-wpLoose"),
        cutbasedID_veto = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Summer16-80X-V1-veto"),
        cutbasedID_loose = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Summer16-80X-V1-loose"),
        cutbasedID_medium = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Summer16-80X-V1-medium"),
        cutbasedID_tight = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Summer16-80X-V1-tight"),
        cutbasedID_HLT = cms.InputTag("egmGsfElectronIDs:cutBasedElectronHLTPreselection-Summer16-V1"),
    ),
    userInts = cms.PSet(
        VIDNestedWPBitmap = cms.InputTag("bitmapVIDForEle"),
    ),
    userCands = cms.PSet(
        jetForLepJetVar = cms.InputTag("ptRatioRelForEle:jetForLepJetVar") # warning: Ptr is null if no match is found
    ),
)

# this below is used only in some eras
slimmedElectronsWithDZ = cms.EDProducer("PATElectronUpdater",
    src = cms.InputTag("slimmedElectronsWithUserData"),
    vertices = cms.InputTag("offlineSlimmedPrimaryVertices")
)

finalElectrons = cms.EDFilter("PATElectronRefSelector",
    src = cms.InputTag("slimmedElectronsWithUserData"),
    cut = cms.string("pt > 5 ")
)
run2_miniAOD_80XLegacy.toModify(finalElectrons, src = "slimmedElectronsWithDZ")
run2_nanoAOD_92X.toModify(finalElectrons, src = "slimmedElectronsWithDZ")

electronMVATTH= cms.EDProducer("EleBaseMVAValueMapProducer",
    src = cms.InputTag("linkedObjects","electrons"),
    weightFile =  cms.FileInPath("PhysicsTools/NanoAOD/data/el_BDTG.weights.xml"),
    name = cms.string("electronMVATTH"),
    isClassifier = cms.bool(True),
    variablesOrder = cms.vstring(["LepGood_pt","LepGood_eta","LepGood_jetNDauChargedMVASel","LepGood_miniRelIsoCharged","LepGood_miniRelIsoNeutral","LepGood_jetPtRelv2","LepGood_jetPtRatio","LepGood_jetBTagCSV","LepGood_sip3d","LepGood_dxy","LepGood_dz","LepGood_mvaIdSpring16HZZ"]),
    variables = cms.PSet(
        LepGood_pt = cms.string("pt"),
        LepGood_eta = cms.string("eta"),
        LepGood_jetNDauChargedMVASel = cms.string("userFloat('jetNDauChargedMVASel')"),
        LepGood_miniRelIsoCharged = cms.string("userFloat('miniIsoChg')/pt"),
        LepGood_miniRelIsoNeutral = cms.string("(userFloat('miniIsoAll')-userFloat('miniIsoChg'))/pt"),
        LepGood_jetPtRelv2 = cms.string("userFloat('ptRel')"),
        LepGood_jetPtRatio = cms.string("min(userFloat('ptRatio'),1.5)"),
        LepGood_jetBTagCSV = cms.string("?userCand('jetForLepJetVar').isNonnull()?max(userCand('jetForLepJetVar').bDiscriminator('pfCombinedInclusiveSecondaryVertexV2BJetTags'),0.0):-99.0"),
        LepGood_sip3d = cms.string("abs(dB('PV3D')/edB('PV3D'))"),
        LepGood_dxy = cms.string("log(abs(dB('PV2D')))"),
        LepGood_dz = cms.string("log(abs(dB('PVDZ')))"),
        LepGood_mvaIdSpring16HZZ = cms.string("userFloat('mvaSpring16HZZ')"),
    )
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
        #ptErr = Var("gsfTrack().ptError()",float,doc="pt error of the GSF track",precision=6),
        energyErr = Var("p4Error('P4_COMBINATION')*userFloat('eCorr')",float,doc="energy error of the cluster-track combination",precision=6),
        eCorr = Var("userFloat('eCorr')",float,doc="ratio of the calibrated energy/miniaod energy"),
        dz = Var("dB('PVDZ')",float,doc="dz (with sign) wrt first PV, in cm",precision=10),
        dzErr = Var("abs(edB('PVDZ'))",float,doc="dz uncertainty, in cm",precision=6),
        dxy = Var("dB('PV2D')",float,doc="dxy (with sign) wrt first PV, in cm",precision=10),
        dxyErr = Var("edB('PV2D')",float,doc="dxy uncertainty, in cm",precision=6),
        ip3d = Var("abs(dB('PV3D'))",float,doc="3D impact parameter wrt first PV, in cm",precision=10),
        sip3d = Var("abs(dB('PV3D')/edB('PV3D'))",float,doc="3D impact parameter significance wrt first PV, in cm",precision=10),
        deltaEtaSC = Var("superCluster().eta()-eta()",float,doc="delta eta (SC,ele) with sign",precision=10),
        r9 = Var("full5x5_r9()",float,doc="R9 of the supercluster, calculated with full 5x5 region",precision=10),
        sieie = Var("full5x5_sigmaIetaIeta()",float,doc="sigma_IetaIeta of the supercluster, calculated with full 5x5 region",precision=10),
        mvaSpring16GP = Var("userFloat('mvaSpring16GP')",float,doc="MVA general-purpose ID score"),
        mvaSpring16GP_WP80 = Var("userInt('mvaSpring16GP_WP80')",bool,doc="MVA general-purpose ID WP80"),
        mvaSpring16GP_WP90 = Var("userInt('mvaSpring16GP_WP90')",bool,doc="MVA general-purpose ID WP90"),
        mvaSpring16HZZ = Var("userFloat('mvaSpring16HZZ')",float,doc="MVA HZZ ID score"),
        mvaSpring16HZZ_WPL = Var("userInt('mvaSpring16HZZ_WPL')",bool,doc="MVA HZZ ID loose WP"),
        cutBased = Var("userInt('cutbasedID_veto')+userInt('cutbasedID_loose')+userInt('cutbasedID_medium')+userInt('cutbasedID_tight')",int,doc="cut-based ID (0:fail, 1:veto, 2:loose, 3:medium, 4:tight)"),
        vidNestedWPBitmap = Var("userInt('VIDNestedWPBitmap')",int,doc=_bitmapVIDForEle_docstring),
        cutBased_HLTPreSel = Var("userInt('cutbasedID_HLT')",int,doc="cut-based HLT pre-selection ID"),
        miniPFRelIso_chg = Var("userFloat('miniIsoChg')/pt",float,doc="mini PF relative isolation, charged component"),
        miniPFRelIso_all = Var("userFloat('miniIsoAll')/pt",float,doc="mini PF relative isolation, total (with scaled rho*EA PU corrections)"),
        pfRelIso03_chg = Var("userFloat('PFIsoChg')/pt",float,doc="PF relative isolation dR=0.3, charged component"),
        pfRelIso03_all = Var("userFloat('PFIsoAll')/pt",float,doc="PF relative isolation dR=0.3, total (with rho*EA PU corrections)"),
        hoe = Var("hadronicOverEm()",float,doc="H over E",precision=8),
        tightCharge = Var("isGsfCtfScPixChargeConsistent() + isGsfScPixChargeConsistent()",int,doc="Tight charge criteria (0:none, 1:isGsfScPixChargeConsistent, 2:isGsfCtfScPixChargeConsistent)"),
        convVeto = Var("passConversionVeto()",bool,doc="pass conversion veto"),
        lostHits = Var("gsfTrack.hitPattern.numberOfLostHits('MISSING_INNER_HITS')","uint8",doc="number of missing inner hits"),
    ),
    externalVariables = cms.PSet(
        mvaTTH = ExtVar(cms.InputTag("electronMVATTH"),float, doc="TTH MVA lepton ID score",precision=14),
    ),
)
electronTable.variables.pt = Var("pt*userFloat('eCorr')",  float, precision=-1)

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

electronSequence = cms.Sequence(egmGsfElectronIDSequence + bitmapVIDForEle + isoForEle + ptRatioRelForEle + calibratedPatElectrons + energyCorrForEle + slimmedElectronsWithUserData + finalElectrons)
electronTables = cms.Sequence (electronMVATTH + electronTable)
electronMC = cms.Sequence(electronsMCMatchForTable + electronMCTable)

_withDZ_sequence = electronSequence.copy()
_withDZ_sequence.replace(finalElectrons, slimmedElectronsWithDZ+finalElectrons)
run2_nanoAOD_92X.toReplaceWith(electronSequence, _withDZ_sequence)
run2_miniAOD_80XLegacy.toReplaceWith(electronSequence, _withDZ_sequence)
