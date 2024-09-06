import FWCore.ParameterSet.Config as cms

from PhysicsTools.NanoAOD.nano_eras_cff import *
from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.simplePATMuonFlatTableProducer_cfi import simplePATMuonFlatTableProducer

import PhysicsTools.PatAlgos.producersLayer1.muonProducer_cfi

# this below is used only in some eras
slimmedMuonsUpdated = cms.EDProducer("PATMuonUpdater",
    src = cms.InputTag("slimmedMuons"),
    vertices = cms.InputTag("offlineSlimmedPrimaryVertices"),
    computeMiniIso = cms.bool(False),
    fixDxySign = cms.bool(True),
    pfCandsForMiniIso = cms.InputTag("packedPFCandidates"),
    miniIsoParams = PhysicsTools.PatAlgos.producersLayer1.muonProducer_cfi.patMuons.miniIsoParams, # so they're in sync
    recomputeMuonBasicSelectors = cms.bool(False),
)

isoForMu = cms.EDProducer("MuonIsoValueMapProducer",
    src = cms.InputTag("slimmedMuonsUpdated"),
    relative = cms.bool(False),
    rho_MiniIso = cms.InputTag("fixedGridRhoFastjetAll"),
    EAFile_MiniIso = cms.FileInPath("PhysicsTools/NanoAOD/data/effAreaMuons_cone03_pfNeuHadronsAndPhotons_94X.txt"),
)

ptRatioRelForMu = cms.EDProducer("MuonJetVarProducer",
    srcJet = cms.InputTag("updatedJetsPuppi"),
    srcLep = cms.InputTag("slimmedMuonsUpdated"),
    srcVtx = cms.InputTag("offlineSlimmedPrimaryVertices"),
)

muonMVAID = cms.EDProducer("EvaluateMuonMVAID",
    src = cms.InputTag("slimmedMuonsUpdated"),
    weightFile =  cms.FileInPath("RecoMuon/MuonIdentification/data/mvaID.onnx"),
    backend = cms.string('ONNX'),
    name = cms.string("muonMVAID"),
    outputTensorName= cms.string("probabilities"),
    inputTensorName= cms.string("float_input"),
    outputNames = cms.vstring(["probGOOD", "wpMedium", "wpTight"]),
    batch_eval =cms.bool(True),
    outputFormulas = cms.vstring(["at(1)", "? at(1) > 0.08 ? 1 : 0", "? at(1) > 0.20 ? 1 : 0"]),
    variables = cms.VPSet(
        cms.PSet( name = cms.string("LepGood_global_muon"), expr = cms.string("isGlobalMuon")),
        cms.PSet( name = cms.string("LepGood_validFraction"), expr = cms.string("?innerTrack.isNonnull?innerTrack().validFraction:-99")),
        cms.PSet( name = cms.string("Muon_norm_chi2_extended")),
        cms.PSet( name = cms.string("LepGood_local_chi2"), expr = cms.string("combinedQuality().chi2LocalPosition")),
        cms.PSet( name = cms.string("LepGood_kink"), expr = cms.string("combinedQuality().trkKink")),
        cms.PSet( name = cms.string("LepGood_segmentComp"), expr = cms.string("segmentCompatibility")),
        cms.PSet( name = cms.string("Muon_n_Valid_hits_extended")),
        cms.PSet( name = cms.string("LepGood_n_MatchedStations"), expr = cms.string("numberOfMatchedStations()")),
        cms.PSet( name = cms.string("LepGood_Valid_pixel"), expr = cms.string("?innerTrack.isNonnull()?innerTrack().hitPattern().numberOfValidPixelHits():-99")),
        cms.PSet( name = cms.string("LepGood_tracker_layers"), expr = cms.string("?innerTrack.isNonnull()?innerTrack().hitPattern().trackerLayersWithMeasurement():-99")),
        cms.PSet( name = cms.string("LepGood_pt"), expr = cms.string("pt")),
        cms.PSet( name = cms.string("LepGood_eta"), expr = cms.string("eta")),
    )
)






slimmedMuonsWithUserData = cms.EDProducer("PATMuonUserDataEmbedder",
     src = cms.InputTag("slimmedMuonsUpdated"),
     userFloats = cms.PSet(
        miniIsoChg = cms.InputTag("isoForMu:miniIsoChg"),
        miniIsoAll = cms.InputTag("isoForMu:miniIsoAll"),
        ptRatio = cms.InputTag("ptRatioRelForMu:ptRatio"),
        ptRel = cms.InputTag("ptRatioRelForMu:ptRel"),
        jetNDauChargedMVASel = cms.InputTag("ptRatioRelForMu:jetNDauChargedMVASel"),
        mvaIDMuon_wpMedium = cms.InputTag("muonMVAID:wpMedium"),
        mvaIDMuon_wpTight = cms.InputTag("muonMVAID:wpTight"),
        mvaIDMuon = cms.InputTag("muonMVAID:probGOOD")
     ),
     userCands = cms.PSet(
        jetForLepJetVar = cms.InputTag("ptRatioRelForMu:jetForLepJetVar") # warning: Ptr is null if no match is found
     ),
)


finalMuons = cms.EDFilter("PATMuonRefSelector",
    src = cms.InputTag("slimmedMuonsWithUserData"),
    cut = cms.string("pt > 15 || (pt > 3 && (passed('CutBasedIdLoose') || passed('SoftCutBasedId') || passed('SoftMvaId') || passed('CutBasedIdGlobalHighPt') || passed('CutBasedIdTrkHighPt')))")
)

finalLooseMuons = cms.EDFilter("PATMuonRefSelector", # for isotrack cleaning
    src = cms.InputTag("slimmedMuonsWithUserData"),
    cut = cms.string("pt > 3 && track.isNonnull && isLooseMuon")
)

muonPROMPTMVA= cms.EDProducer("MuonBaseMVAValueMapProducer",
    src = cms.InputTag("linkedObjects","muons"),
    weightFile =  cms.FileInPath("PhysicsTools/NanoAOD/data/mu_BDTG_2022.weights.xml"),
    backend = cms.string("TMVA"),
    name = cms.string("muonPROMPTMVA"),
    isClassifier = cms.bool(True),
    variables = cms.VPSet(
        cms.PSet( name = cms.string("LepGood_pt"), expr = cms.string("pt")),
        cms.PSet( name = cms.string("LepGood_eta"), expr = cms.string("eta")),
        cms.PSet( name = cms.string("LepGood_pfRelIso03_all"), expr = cms.string("(pfIsolationR03().sumChargedHadronPt + max(pfIsolationR03().sumNeutralHadronEt + pfIsolationR03().sumPhotonEt - pfIsolationR03().sumPUPt/2,0.0))/pt")),
        cms.PSet( name = cms.string("LepGood_miniRelIsoCharged"), expr = cms.string("userFloat('miniIsoChg')/pt")),
        cms.PSet( name = cms.string("LepGood_miniRelIsoNeutral"), expr = cms.string("(userFloat('miniIsoAll')-userFloat('miniIsoChg'))/pt")),
        cms.PSet( name = cms.string("LepGood_jetNDauChargedMVASel"), expr = cms.string("?userCand('jetForLepJetVar').isNonnull()?userFloat('jetNDauChargedMVASel'):0")),
        cms.PSet( name = cms.string("LepGood_jetPtRelv2"), expr = cms.string("?userCand('jetForLepJetVar').isNonnull()?userFloat('ptRel'):0")),
        cms.PSet( name = cms.string("LepGood_jetDF"), expr = cms.string("?userCand('jetForLepJetVar').isNonnull()?max(userCand('jetForLepJetVar').bDiscriminator('pfDeepFlavourJetTags:probbb')+userCand('jetForLepJetVar').bDiscriminator('pfDeepFlavourJetTags:probb')+userCand('jetForLepJetVar').bDiscriminator('pfDeepFlavourJetTags:problepb'),0.0):0.0")),
        cms.PSet( name = cms.string("LepGood_jetPtRatio"), expr = cms.string("?userCand('jetForLepJetVar').isNonnull()?min(userFloat('ptRatio'),1.5):1.0/(1.0+(pfIsolationR04().sumChargedHadronPt + max(pfIsolationR04().sumNeutralHadronEt + pfIsolationR04().sumPhotonEt - pfIsolationR04().sumPUPt/2,0.0))/pt)")),
        cms.PSet( name = cms.string("LepGood_sip3d"), expr = cms.string("abs(dB('PV3D')/edB('PV3D'))")),
        cms.PSet( name = cms.string("LepGood_dxy"), expr = cms.string("log(abs(dB('PV2D')))")),
        cms.PSet( name = cms.string("LepGood_dz"), expr = cms.string("log(abs(dB('PVDZ')))")),
        cms.PSet( name = cms.string("LepGood_segmentComp"), expr = cms.string("segmentCompatibility")),
        )
)

_legacy_muon_BDT_variable = cms.VPSet(
    cms.PSet( name = cms.string("LepGood_pt"), expr = cms.string("pt")),
    cms.PSet( name = cms.string("LepGood_eta"), expr = cms.string("eta")),
    cms.PSet( name = cms.string("LepGood_jetNDauChargedMVASel"), expr = cms.string("?userCand('jetForLepJetVar').isNonnull()?userFloat('jetNDauChargedMVASel'):0")),
    cms.PSet( name = cms.string("LepGood_miniRelIsoCharged"), expr = cms.string("userFloat('miniIsoChg')/pt")),
    cms.PSet( name = cms.string("LepGood_miniRelIsoNeutral"), expr = cms.string("(userFloat('miniIsoAll')-userFloat('miniIsoChg'))/pt")),
    cms.PSet( name = cms.string("LepGood_jetPtRelv2"), expr = cms.string("?userCand('jetForLepJetVar').isNonnull()?userFloat('ptRel'):0")),
    cms.PSet( name = cms.string("LepGood_jetDF"), expr = cms.string("?userCand('jetForLepJetVar').isNonnull()?max(userCand('jetForLepJetVar').bDiscriminator('pfDeepFlavourJetTags:probbb')+userCand('jetForLepJetVar').bDiscriminator('pfDeepFlavourJetTags:probb')+userCand('jetForLepJetVar').bDiscriminator('pfDeepFlavourJetTags:problepb'),0.0):0.0")),
    cms.PSet( name = cms.string("LepGood_jetPtRatio"), expr = cms.string("?userCand('jetForLepJetVar').isNonnull()?min(userFloat('ptRatio'),1.5):1.0/(1.0+(pfIsolationR04().sumChargedHadronPt + max(pfIsolationR04().sumNeutralHadronEt + pfIsolationR04().sumPhotonEt - pfIsolationR04().sumPUPt/2,0.0))/pt)")),
    cms.PSet( name = cms.string("LepGood_dxy"), expr = cms.string("log(abs(dB('PV2D')))")),
    cms.PSet( name = cms.string("LepGood_sip3d"), expr = cms.string("abs(dB('PV3D')/edB('PV3D'))")),
    cms.PSet( name = cms.string("LepGood_dz"), expr = cms.string("log(abs(dB('PVDZ')))")),
    cms.PSet( name = cms.string("LepGood_segmentComp"), expr = cms.string("segmentCompatibility")),
)

muonMVALowPt = muonPROMPTMVA.clone(
    weightFile =  cms.FileInPath("PhysicsTools/NanoAOD/data/mu_BDTG_lowpt.weights.xml"),
    name = cms.string("muonMVALowPt"),
    variables = _legacy_muon_BDT_variable
)

run2_muon_2016.toModify(
    muonPROMPTMVA,
    weightFile = "PhysicsTools/NanoAOD/data/mu_BDTG_2016.weights.xml",
    variables =	_legacy_muon_BDT_variable
)

(run2_muon_2017 | run2_muon_2018).toModify(
    muonPROMPTMVA,
    weightFile =  cms.FileInPath("PhysicsTools/NanoAOD/data/mu_BDTG_2017.weights.xml"),
    variables = _legacy_muon_BDT_variable
)

from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *
muonBSConstrain = cms.EDProducer("MuonBeamspotConstraintValueMapProducer",
    src = cms.InputTag("linkedObjects","muons"),
)

muonTable = simplePATMuonFlatTableProducer.clone(
    src = cms.InputTag("linkedObjects","muons"),
    name = cms.string("Muon"),
    doc  = cms.string("slimmedMuons after basic selection (" + finalMuons.cut.value()+")"),
    variables = cms.PSet(CandVars,
        ptErr   = Var("bestTrack().ptError()", float, doc = "ptError of the muon track", precision=6),
        tunepRelPt = Var("tunePMuonBestTrack().pt/pt",float,doc="TuneP relative pt, tunePpt/pt",precision=6),
        tuneP_pterr = Var("tunePMuonBestTrack().ptError()", float, doc = "pTerr from tunePMuonBestTrack", precision=6),
        dz = Var("dB('PVDZ')",float,doc="dz (with sign) wrt first PV, in cm",precision=10),
        dzErr = Var("abs(edB('PVDZ'))",float,doc="dz uncertainty, in cm",precision=6),
        dxybs = Var("dB('BS2D')",float,doc="dxy (with sign) wrt the beam spot, in cm",precision=10),
        dxy = Var("dB('PV2D')",float,doc="dxy (with sign) wrt first PV, in cm",precision=10),
        dxyErr = Var("edB('PV2D')",float,doc="dxy uncertainty, in cm",precision=6),
        ip3d = Var("abs(dB('PV3D'))",float,doc="3D impact parameter wrt first PV, in cm",precision=10),
        sip3d = Var("abs(dB('PV3D')/edB('PV3D'))",float,doc="3D impact parameter significance wrt first PV",precision=10),
        segmentComp   = Var("segmentCompatibility()", float, doc = "muon segment compatibility", precision=14), # keep higher precision since people have cuts with 3 digits on this
        nStations = Var("numberOfMatchedStations", "uint8", doc = "number of matched stations with default arbitration (segment & track)"),
        nTrackerLayers = Var("?track.isNonnull?innerTrack().hitPattern().trackerLayersWithMeasurement():0", "uint8", doc = "number of layers in the tracker"),
        highPurity = Var("?track.isNonnull?innerTrack().quality('highPurity'):0", bool, doc = "inner track is high purity"),
        jetIdx = Var("?hasUserCand('jet')?userCand('jet').key():-1", "int16", doc="index of the associated jet (-1 if none)"),
        svIdx = Var("?hasUserCand('vertex')?userCand('vertex').key():-1", "int16", doc="index of matching secondary vertex"),
        tkRelIso = Var("isolationR03().sumPt/pt",float,doc="Tracker-based relative isolation dR=0.3 for highPt, trkIso/pt",precision=6),
        miniPFRelIso_chg = Var("userFloat('miniIsoChg')/pt",float,doc="mini PF relative isolation, charged component"),
        miniPFRelIso_all = Var("userFloat('miniIsoAll')/pt",float,doc="mini PF relative isolation, total (with scaled rho*EA PU corrections)"),
        pfRelIso03_chg = Var("pfIsolationR03().sumChargedHadronPt/pt",float,doc="PF relative isolation dR=0.3, charged component"),
        pfRelIso03_all = Var("(pfIsolationR03().sumChargedHadronPt + max(pfIsolationR03().sumNeutralHadronEt + pfIsolationR03().sumPhotonEt - pfIsolationR03().sumPUPt/2,0.0))/pt",float,doc="PF relative isolation dR=0.3, total (deltaBeta corrections)"),
        pfRelIso04_all = Var("(pfIsolationR04().sumChargedHadronPt + max(pfIsolationR04().sumNeutralHadronEt + pfIsolationR04().sumPhotonEt - pfIsolationR04().sumPUPt/2,0.0))/pt",float,doc="PF relative isolation dR=0.4, total (deltaBeta corrections)"),
        jetRelIso = Var("?userCand('jetForLepJetVar').isNonnull()?(1./userFloat('ptRatio'))-1.:-1.",float,doc="Relative isolation in matched jet (1/ptRatio-1), -1 if none",precision=8),
        jetPtRelv2 = Var("?userCand('jetForLepJetVar').isNonnull()?userFloat('ptRel'):0",float,doc="Relative momentum of the lepton with respect to the closest jet after subtracting the lepton",precision=8),
        jetDF = Var("?userCand('jetForLepJetVar').isNonnull()?max(userCand('jetForLepJetVar').bDiscriminator('pfDeepFlavourJetTags:probbb')+userCand('jetForLepJetVar').bDiscriminator('pfDeepFlavourJetTags:probb')+userCand('jetForLepJetVar').bDiscriminator('pfDeepFlavourJetTags:problepb'),0.0):0.0",float,doc="value of the DEEPJET b tagging algorithm discriminator of the associated jet (0 if none)",precision=8,lazyEval=True),
        tightCharge = Var("?(muonBestTrack().ptError()/muonBestTrack().pt() < 0.2)?2:0", "uint8", doc="Tight charge criterion using pterr/pt of muonBestTrack (0:fail, 2:pass)"),
        looseId  = Var("passed('CutBasedIdLoose')",bool, doc="muon is loose muon"),
        isPFcand = Var("isPFMuon",bool,doc="muon is PF candidate"),
        isGlobal = Var("isGlobalMuon",bool,doc="muon is global muon"),
        isTracker = Var("isTrackerMuon",bool,doc="muon is tracker muon"),
        isStandalone = Var("isStandAloneMuon",bool,doc="muon is a standalone muon"),
        mediumId = Var("passed('CutBasedIdMedium')",bool,doc="cut-based ID, medium WP"),
        mediumPromptId = Var("passed('CutBasedIdMediumPrompt')",bool,doc="cut-based ID, medium prompt WP"),
        tightId = Var("passed('CutBasedIdTight')",bool,doc="cut-based ID, tight WP"),
        softId = Var("passed('SoftCutBasedId')",bool,doc="soft cut-based ID"),
        softMvaId = Var("passed('SoftMvaId')",bool,doc="soft MVA ID"),
        softMva = Var("softMvaValue()",float,doc="soft MVA ID score",precision=6),
        softMvaRun3 = Var("softMvaRun3Value()",float,doc="soft MVA Run3 ID score",precision=6),
        highPtId = Var("?passed('CutBasedIdGlobalHighPt')?2:passed('CutBasedIdTrkHighPt')","uint8",doc="high-pT cut-based ID (1 = tracker high pT, 2 = global high pT, which includes tracker high pT)"),
        pfIsoId = Var("passed('PFIsoVeryLoose')+passed('PFIsoLoose')+passed('PFIsoMedium')+passed('PFIsoTight')+passed('PFIsoVeryTight')+passed('PFIsoVeryVeryTight')","uint8",doc="PFIso ID from miniAOD selector (1=PFIsoVeryLoose, 2=PFIsoLoose, 3=PFIsoMedium, 4=PFIsoTight, 5=PFIsoVeryTight, 6=PFIsoVeryVeryTight)"),
        tkIsoId = Var("?passed('TkIsoTight')?2:passed('TkIsoLoose')","uint8",doc="TkIso ID (1=TkIsoLoose, 2=TkIsoTight)"),
        miniIsoId = Var("passed('MiniIsoLoose')+passed('MiniIsoMedium')+passed('MiniIsoTight')+passed('MiniIsoVeryTight')","uint8",doc="MiniIso ID from miniAOD selector (1=MiniIsoLoose, 2=MiniIsoMedium, 3=MiniIsoTight, 4=MiniIsoVeryTight)"),
        mvaMuID = Var("userFloat('mvaIDMuon')", float, doc="MVA-based ID score",precision=6),
        mvaMuID_WP = Var("userFloat('mvaIDMuon_wpMedium') + userFloat('mvaIDMuon_wpTight')","uint8",doc="MVA-based ID selector WPs (1=MVAIDwpMedium,2=MVAIDwpTight)"),
        multiIsoId = Var("?passed('MultiIsoMedium')?2:passed('MultiIsoLoose')","uint8",doc="MultiIsoId from miniAOD selector (1=MultiIsoLoose, 2=MultiIsoMedium)"),
        puppiIsoId = Var("passed('PuppiIsoLoose')+passed('PuppiIsoMedium')+passed('PuppiIsoTight')", "uint8", doc="PuppiIsoId from miniAOD selector (1=Loose, 2=Medium, 3=Tight)"),
        triggerIdLoose = Var("passed('TriggerIdLoose')",bool,doc="TriggerIdLoose ID"),
        inTimeMuon = Var("passed('InTimeMuon')",bool,doc="inTimeMuon ID"),
        jetNDauCharged = Var("?userCand('jetForLepJetVar').isNonnull()?userFloat('jetNDauChargedMVASel'):0", "uint8", doc="number of charged daughters of the closest jet"),
        ),
    externalVariables = cms.PSet(
        promptMVA = ExtVar(cms.InputTag("muonPROMPTMVA"),float, doc="Prompt MVA lepton ID score. Corresponds to the previous mvaTTH",precision=14),
        mvaLowPt = ExtVar(cms.InputTag("muonMVALowPt"),float, doc="Low pt muon ID score",precision=14),
        fsrPhotonIdx = ExtVar(cms.InputTag("leptonFSRphotons:muFsrIndex"), "int16", doc="Index of the lowest-dR/ET2 among associated FSR photons"),
        bsConstrainedPt = ExtVar(cms.InputTag("muonBSConstrain:muonBSConstrainedPt"),float, doc="pT with beamspot constraint",precision=-1),
        bsConstrainedPtErr = ExtVar(cms.InputTag("muonBSConstrain:muonBSConstrainedPtErr"),float, doc="pT error with beamspot constraint ",precision=6),
        bsConstrainedChi2 = ExtVar(cms.InputTag("muonBSConstrain:muonBSConstrainedChi2"),float, doc="chi2 of beamspot constraint",precision=6),
    ),
)

# Increase precision of eta and phi
muonTable.variables.eta.precision = 16
muonTable.variables.phi.precision = 16


# Revert back to AK4 CHS jets for Run 2
run2_nanoAOD_ANY.toModify(
    ptRatioRelForMu,srcJet="updatedJets"
)


muonsMCMatchForTable = cms.EDProducer("MCMatcher",       # cut on deltaR, deltaPt/Pt; pick best by deltaR
    src         = muonTable.src,                         # final reco collection
    matched     = cms.InputTag("finalGenParticles"),     # final mc-truth particle collection
    mcPdgId     = cms.vint32(13),               # one or more PDG ID (13 = mu); absolute values (see below)
    checkCharge = cms.bool(False),              # True = require RECO and MC objects to have the same charge
    mcStatus    = cms.vint32(1),                # PYTHIA status code (1 = stable, 2 = shower, 3 = hard scattering)
    maxDeltaR   = cms.double(0.3),              # Minimum deltaR for the match
    maxDPtRel   = cms.double(0.5),              # Minimum deltaPt/Pt for the match
    resolveAmbiguities    = cms.bool(True),     # Forbid two RECO objects to match to the same GEN object
    resolveByMatchQuality = cms.bool(True),    # False = just match input in order; True = pick lowest deltaR pair first
)

muonMCTable = cms.EDProducer("CandMCMatchTableProducer",
    src     = muonTable.src,
    mcMap   = cms.InputTag("muonsMCMatchForTable"),
    objName = muonTable.name,
    objType = muonTable.name, #cms.string("Muon"),
    branchName = cms.string("genPart"),
    docString = cms.string("MC matching to status==1 muons"),
)

muonTask = cms.Task(slimmedMuonsUpdated,isoForMu,ptRatioRelForMu,slimmedMuonsWithUserData,finalMuons,finalLooseMuons )
muonMCTask = cms.Task(muonsMCMatchForTable,muonMCTable)
muonTablesTask = cms.Task(muonPROMPTMVA,muonMVALowPt,muonBSConstrain,muonTable,muonMVAID)

