import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Modifier_run2_miniAOD_80XLegacy_cff import run2_miniAOD_80XLegacy
from Configuration.Eras.Modifier_run2_nanoAOD_92X_cff import run2_nanoAOD_92X
from PhysicsTools.NanoAOD.common_cff import *

isoForMu = cms.EDProducer("MuonIsoValueMapProducer",
    src = cms.InputTag("slimmedMuons"),
    relative = cms.bool(False),
    rho_MiniIso = cms.InputTag("fixedGridRhoFastjetCentralNeutral"),
    EAFile_MiniIso = cms.FileInPath("PhysicsTools/NanoAOD/data/effAreaMuons_cone03_pfNeuHadronsAndPhotons_80X.txt"),
)

ptRatioRelForMu = cms.EDProducer("MuonJetVarProducer",
    srcJet = cms.InputTag("slimmedJets"),
    srcLep = cms.InputTag("slimmedMuons"),
    srcVtx = cms.InputTag("offlineSlimmedPrimaryVertices"),
)

slimmedMuonsWithUserData = cms.EDProducer("PATMuonUserDataEmbedder",
     src = cms.InputTag("slimmedMuons"),
     userFloats = cms.PSet(
        miniIsoChg = cms.InputTag("isoForMu:miniIsoChg"),
        miniIsoAll = cms.InputTag("isoForMu:miniIsoAll"),
        ptRatio = cms.InputTag("ptRatioRelForMu:ptRatio"),
        ptRel = cms.InputTag("ptRatioRelForMu:ptRel"),
        jetNDauChargedMVASel = cms.InputTag("ptRatioRelForMu:jetNDauChargedMVASel"),
     ),
     userCands = cms.PSet(
        jetForLepJetVar = cms.InputTag("ptRatioRelForMu:jetForLepJetVar") # warning: Ptr is null if no match is found
     ),
)
# this below is used only in some eras
slimmedMuonsWithDZ = cms.EDProducer("PATMuonUpdater",
    src = cms.InputTag("slimmedMuonsWithUserData"),
    vertices = cms.InputTag("offlineSlimmedPrimaryVertices")
)

finalMuons = cms.EDFilter("PATMuonRefSelector",
    src = cms.InputTag("slimmedMuonsWithUserData"),
    cut = cms.string("pt > 3 && track.isNonnull && isLooseMuon")
)
run2_miniAOD_80XLegacy.toModify(finalMuons, src = "slimmedMuonsWithDZ")
run2_nanoAOD_92X.toModify(finalMuons, src = "slimmedMuonsWithDZ")

muonMVATTH= cms.EDProducer("MuonBaseMVAValueMapProducer",
    src = cms.InputTag("linkedObjects","muons"),
    weightFile =  cms.FileInPath("PhysicsTools/NanoAOD/data/mu_BDTG.weights.xml"),
    name = cms.string("muonMVATTH"),
    isClassifier = cms.bool(True),
    variablesOrder = cms.vstring(["LepGood_pt","LepGood_eta","LepGood_jetNDauChargedMVASel","LepGood_miniRelIsoCharged","LepGood_miniRelIsoNeutral","LepGood_jetPtRelv2","LepGood_jetPtRatio","LepGood_jetBTagCSV","LepGood_sip3d","LepGood_dxy","LepGood_dz","LepGood_segmentCompatibility"]),
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
        LepGood_segmentCompatibility = cms.string("segmentCompatibility"),
    )
)

muonTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
    src = cms.InputTag("linkedObjects","muons"),
    cut = cms.string(""), #we should not filter on cross linked collections
    name = cms.string("Muon"),
    doc  = cms.string("slimmedMuons after basic selection (" + finalMuons.cut.value()+")"),
    singleton = cms.bool(False), # the number of entries is variable
    extension = cms.bool(False), # this is the main table for the muons
    variables = cms.PSet(CandVars,
        ptErr   = Var("bestTrack().ptError()", float, doc = "ptError of the muon track", precision=6),
        dz = Var("dB('PVDZ')",float,doc="dz (with sign) wrt first PV, in cm",precision=10),
        dzErr = Var("abs(edB('PVDZ'))",float,doc="dz uncertainty, in cm",precision=6),
        dxy = Var("dB('PV2D')",float,doc="dxy (with sign) wrt first PV, in cm",precision=10),
        dxyErr = Var("edB('PV2D')",float,doc="dxy uncertainty, in cm",precision=6),
        ip3d = Var("abs(dB('PV3D'))",float,doc="3D impact parameter wrt first PV, in cm",precision=10),
        sip3d = Var("abs(dB('PV3D')/edB('PV3D'))",float,doc="3D impact parameter significance wrt first PV",precision=10),
        segmentComp   = Var("segmentCompatibility()", float, doc = "muon segment compatibility", precision=14), # keep higher precision since people have cuts with 3 digits on this
        nStations = Var("numberOfMatchedStations", int, doc = "number of matched stations with default arbitration (segment & track)"),
        jetIdx = Var("?hasUserCand('jet')?userCand('jet').key():-1", int, doc="index of the associated jet (-1 if none)"),
        miniPFRelIso_chg = Var("userFloat('miniIsoChg')/pt",float,doc="mini PF relative isolation, charged component"),
        miniPFRelIso_all = Var("userFloat('miniIsoAll')/pt",float,doc="mini PF relative isolation, total (with scaled rho*EA PU corrections)"),
        pfRelIso03_chg = Var("pfIsolationR03().sumChargedHadronPt/pt",float,doc="PF relative isolation dR=0.3, charged component"),
        pfRelIso03_all = Var("(pfIsolationR03().sumChargedHadronPt + max(pfIsolationR03().sumNeutralHadronEt + pfIsolationR03().sumPhotonEt - pfIsolationR03().sumPUPt/2,0.0))/pt",float,doc="PF relative isolation dR=0.3, total (deltaBeta corrections)"),
        pfRelIso04_all = Var("(pfIsolationR04().sumChargedHadronPt + max(pfIsolationR04().sumNeutralHadronEt + pfIsolationR04().sumPhotonEt - pfIsolationR04().sumPUPt/2,0.0))/pt",float,doc="PF relative isolation dR=0.4, total (deltaBeta corrections)"),
        tightCharge = Var("?(muonBestTrack().ptError()/muonBestTrack().pt() < 0.2)?2:0",int,doc="Tight charge criterion using pterr/pt of muonBestTrack (0:fail, 2:pass)"),
    ),
    externalVariables = cms.PSet(
        mvaTTH = ExtVar(cms.InputTag("muonMVATTH"),float, doc="TTH MVA lepton ID score",precision=14),
    ),
)

muonIDTable = cms.EDProducer("MuonIDTableProducer",
    name = muonTable.name,
    muons = muonTable.src,  # final reco collection
    vertices = cms.InputTag("offlineSlimmedPrimaryVertices"),
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

muonSequence = cms.Sequence(isoForMu + ptRatioRelForMu + slimmedMuonsWithUserData + finalMuons)
muonMC = cms.Sequence(muonsMCMatchForTable + muonMCTable)
muonTables = cms.Sequence(muonMVATTH + muonTable + muonIDTable)

_withDZ_sequence = muonSequence.copy()
_withDZ_sequence.replace(finalMuons, slimmedMuonsWithDZ+finalMuons)
run2_nanoAOD_92X.toReplaceWith(muonSequence, _withDZ_sequence)
run2_miniAOD_80XLegacy.toReplaceWith(muonSequence, _withDZ_sequence)
