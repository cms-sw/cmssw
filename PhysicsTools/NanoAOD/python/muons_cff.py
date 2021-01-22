import FWCore.ParameterSet.Config as cms

from PhysicsTools.NanoAOD.nano_eras_cff import *
from PhysicsTools.NanoAOD.common_cff import *
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
run2_miniAOD_80XLegacy.toModify( slimmedMuonsUpdated, computeMiniIso = True, recomputeMuonBasicSelectors = True )

isoForMu = cms.EDProducer("MuonIsoValueMapProducer",
    src = cms.InputTag("slimmedMuonsUpdated"),
    relative = cms.bool(False),
    rho_MiniIso = cms.InputTag("fixedGridRhoFastjetAll"),
    EAFile_MiniIso = cms.FileInPath("PhysicsTools/NanoAOD/data/effAreaMuons_cone03_pfNeuHadronsAndPhotons_94X.txt"),
)
run2_miniAOD_80XLegacy.toModify(isoForMu, EAFile_MiniIso = "PhysicsTools/NanoAOD/data/effAreaMuons_cone03_pfNeuHadronsAndPhotons_80X.txt")
run2_nanoAOD_94X2016.toModify(isoForMu, EAFile_MiniIso = "PhysicsTools/NanoAOD/data/effAreaMuons_cone03_pfNeuHadronsAndPhotons_80X.txt")

ptRatioRelForMu = cms.EDProducer("MuonJetVarProducer",
    srcJet = cms.InputTag("updatedJets"),
    srcLep = cms.InputTag("slimmedMuonsUpdated"),
    srcVtx = cms.InputTag("offlineSlimmedPrimaryVertices"),
)

slimmedMuonsWithUserData = cms.EDProducer("PATMuonUserDataEmbedder",
     src = cms.InputTag("slimmedMuonsUpdated"),
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

finalMuons = cms.EDFilter("PATMuonRefSelector",
    src = cms.InputTag("slimmedMuonsWithUserData"),
    cut = cms.string("pt > 3 && (passed('CutBasedIdLoose') || passed('SoftCutBasedId') || passed('SoftMvaId') || passed('CutBasedIdGlobalHighPt') || passed('CutBasedIdTrkHighPt'))")
)

finalLooseMuons = cms.EDFilter("PATMuonRefSelector", # for isotrack cleaning
    src = cms.InputTag("slimmedMuonsWithUserData"),
    cut = cms.string("pt > 3 && track.isNonnull && isLooseMuon")
)

muonMVATTH= cms.EDProducer("MuonBaseMVAValueMapProducer",
    src = cms.InputTag("linkedObjects","muons"),
    weightFile =  cms.FileInPath("PhysicsTools/NanoAOD/data/mu_BDTG_2017.weights.xml"),
    name = cms.string("muonMVATTH"),
    isClassifier = cms.bool(True),
    variablesOrder = cms.vstring(["LepGood_pt","LepGood_eta","LepGood_jetNDauChargedMVASel","LepGood_miniRelIsoCharged","LepGood_miniRelIsoNeutral","LepGood_jetPtRelv2","LepGood_jetDF","LepGood_jetPtRatio","LepGood_dxy","LepGood_sip3d","LepGood_dz","LepGood_segmentComp"]),
    variables = cms.PSet(
        LepGood_pt = cms.string("pt"),
        LepGood_eta = cms.string("eta"),
        LepGood_jetNDauChargedMVASel = cms.string("?userCand('jetForLepJetVar').isNonnull()?userFloat('jetNDauChargedMVASel'):0"),
        LepGood_miniRelIsoCharged = cms.string("userFloat('miniIsoChg')/pt"),
        LepGood_miniRelIsoNeutral = cms.string("(userFloat('miniIsoAll')-userFloat('miniIsoChg'))/pt"),
        LepGood_jetPtRelv2 = cms.string("?userCand('jetForLepJetVar').isNonnull()?userFloat('ptRel'):0"),
        LepGood_jetDF = cms.string("?userCand('jetForLepJetVar').isNonnull()?max(userCand('jetForLepJetVar').bDiscriminator('pfDeepFlavourJetTags:probbb')+userCand('jetForLepJetVar').bDiscriminator('pfDeepFlavourJetTags:probb')+userCand('jetForLepJetVar').bDiscriminator('pfDeepFlavourJetTags:problepb'),0.0):0.0"),
        LepGood_jetPtRatio = cms.string("?userCand('jetForLepJetVar').isNonnull()?min(userFloat('ptRatio'),1.5):1.0/(1.0+(pfIsolationR04().sumChargedHadronPt + max(pfIsolationR04().sumNeutralHadronEt + pfIsolationR04().sumPhotonEt - pfIsolationR04().sumPUPt/2,0.0))/pt)"),
        LepGood_dxy = cms.string("log(abs(dB('PV2D')))"),
        LepGood_sip3d = cms.string("abs(dB('PV3D')/edB('PV3D'))"),
        LepGood_dz = cms.string("log(abs(dB('PVDZ')))"),
        LepGood_segmentComp = cms.string("segmentCompatibility"),

    )
)

muonMVALowPt = muonMVATTH.clone(
    weightFile =  cms.FileInPath("PhysicsTools/NanoAOD/data/mu_BDTG_lowpt.weights.xml"),
    name = cms.string("muonMVALowPt"),
)

run2_muon_2016.toModify(muonMVATTH,
        weightFile = "PhysicsTools/NanoAOD/data/mu_BDTG_2016.weights.xml",
    )

from MuonAnalysis.MuonAssociators.muonFSRProducer_cfi import muonFSRProducer
muonFSRphotons = muonFSRProducer.clone(
  packedPFCandidates = cms.InputTag("packedPFCandidates"),
  slimmedElectrons = cms.InputTag("slimmedElectrons"),
  muons = cms.InputTag("linkedObjects","muons"),
)
from MuonAnalysis.MuonAssociators.muonFSRAssociator_cfi import muonFSRAssociator
muonFSRassociation = muonFSRAssociator.clone(
    photons = cms.InputTag("muonFSRphotons"),
    muons = cms.InputTag("linkedObjects","muons"),
)

fsrTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
    src = cms.InputTag("muonFSRphotons"),
    cut = cms.string(""), #we should not filter on cross linked collections
    name = cms.string("FsrPhoton"),
    doc  = cms.string("Final state radiation photons emitted by muons"),
    singleton = cms.bool(False), # the number of entries is variable
    extension = cms.bool(False), # this is the main table for the muons
    variables = cms.PSet(P3Vars,
        relIso03 = Var("userFloat('relIso03')",float,doc="relative isolation in a 0.3 cone without CHS"),
        dROverEt2 = Var("userFloat('dROverEt2')",float,doc="deltaR to associated muon divided by photon et2"),
        muonIdx = Var("?hasUserCand('associatedMuon')?userCand('associatedMuon').key():-1",int, doc="index of associated muon")
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
        tunepRelPt = Var("tunePMuonBestTrack().pt/pt",float,doc="TuneP relative pt, tunePpt/pt",precision=6),
        dz = Var("dB('PVDZ')",float,doc="dz (with sign) wrt first PV, in cm",precision=10),
        dzErr = Var("abs(edB('PVDZ'))",float,doc="dz uncertainty, in cm",precision=6),
        dxybs = Var("dB('BS2D')",float,doc="dxy (with sign) wrt the beam spot, in cm",precision=10),
        dxy = Var("dB('PV2D')",float,doc="dxy (with sign) wrt first PV, in cm",precision=10),
        dxyErr = Var("edB('PV2D')",float,doc="dxy uncertainty, in cm",precision=6),
        ip3d = Var("abs(dB('PV3D'))",float,doc="3D impact parameter wrt first PV, in cm",precision=10),
        sip3d = Var("abs(dB('PV3D')/edB('PV3D'))",float,doc="3D impact parameter significance wrt first PV",precision=10),
        segmentComp   = Var("segmentCompatibility()", float, doc = "muon segment compatibility", precision=14), # keep higher precision since people have cuts with 3 digits on this
        nStations = Var("numberOfMatchedStations", int, doc = "number of matched stations with default arbitration (segment & track)"),
        nTrackerLayers = Var("?track.isNonnull?innerTrack().hitPattern().trackerLayersWithMeasurement():0", int, doc = "number of layers in the tracker"),
        highPurity = Var("?track.isNonnull?innerTrack().quality('highPurity'):0", bool, doc = "inner track is high purity"),
        jetIdx = Var("?hasUserCand('jet')?userCand('jet').key():-1", int, doc="index of the associated jet (-1 if none)"),
        tkRelIso = Var("isolationR03().sumPt/tunePMuonBestTrack().pt",float,doc="Tracker-based relative isolation dR=0.3 for highPt, trkIso/tunePpt",precision=6),
        miniPFRelIso_chg = Var("userFloat('miniIsoChg')/pt",float,doc="mini PF relative isolation, charged component"),
        miniPFRelIso_all = Var("userFloat('miniIsoAll')/pt",float,doc="mini PF relative isolation, total (with scaled rho*EA PU corrections)"),
        pfRelIso03_chg = Var("pfIsolationR03().sumChargedHadronPt/pt",float,doc="PF relative isolation dR=0.3, charged component"),
        pfRelIso03_all = Var("(pfIsolationR03().sumChargedHadronPt + max(pfIsolationR03().sumNeutralHadronEt + pfIsolationR03().sumPhotonEt - pfIsolationR03().sumPUPt/2,0.0))/pt",float,doc="PF relative isolation dR=0.3, total (deltaBeta corrections)"),
        pfRelIso04_all = Var("(pfIsolationR04().sumChargedHadronPt + max(pfIsolationR04().sumNeutralHadronEt + pfIsolationR04().sumPhotonEt - pfIsolationR04().sumPUPt/2,0.0))/pt",float,doc="PF relative isolation dR=0.4, total (deltaBeta corrections)"),
        jetRelIso = Var("?userCand('jetForLepJetVar').isNonnull()?(1./userFloat('ptRatio'))-1.:(pfIsolationR04().sumChargedHadronPt + max(pfIsolationR04().sumNeutralHadronEt + pfIsolationR04().sumPhotonEt - pfIsolationR04().sumPUPt/2,0.0))/pt",float,doc="Relative isolation in matched jet (1/ptRatio-1, pfRelIso04_all if no matched jet)",precision=8),
        jetPtRelv2 = Var("?userCand('jetForLepJetVar').isNonnull()?userFloat('ptRel'):0",float,doc="Relative momentum of the lepton with respect to the closest jet after subtracting the lepton",precision=8),
        tightCharge = Var("?(muonBestTrack().ptError()/muonBestTrack().pt() < 0.2)?2:0",int,doc="Tight charge criterion using pterr/pt of muonBestTrack (0:fail, 2:pass)"),
        looseId  = Var("passed('CutBasedIdLoose')",bool, doc="muon is loose muon"),
        isPFcand = Var("isPFMuon",bool,doc="muon is PF candidate"),
        isGlobal = Var("isGlobalMuon",bool,doc="muon is global muon"),
        isTracker = Var("isTrackerMuon",bool,doc="muon is tracker muon"),
        mediumId = Var("passed('CutBasedIdMedium')",bool,doc="cut-based ID, medium WP"),
        mediumPromptId = Var("passed('CutBasedIdMediumPrompt')",bool,doc="cut-based ID, medium prompt WP"),
        tightId = Var("passed('CutBasedIdTight')",bool,doc="cut-based ID, tight WP"),
        softId = Var("passed('SoftCutBasedId')",bool,doc="soft cut-based ID"),
        softMvaId = Var("passed('SoftMvaId')",bool,doc="soft MVA ID"),
        softMva = Var("softMvaValue()",float,doc="soft MVA ID score",precision=6),
        highPtId = Var("?passed('CutBasedIdGlobalHighPt')?2:passed('CutBasedIdTrkHighPt')","uint8",doc="high-pT cut-based ID (1 = tracker high pT, 2 = global high pT, which includes tracker high pT)"),
        pfIsoId = Var("passed('PFIsoVeryLoose')+passed('PFIsoLoose')+passed('PFIsoMedium')+passed('PFIsoTight')+passed('PFIsoVeryTight')+passed('PFIsoVeryVeryTight')","uint8",doc="PFIso ID from miniAOD selector (1=PFIsoVeryLoose, 2=PFIsoLoose, 3=PFIsoMedium, 4=PFIsoTight, 5=PFIsoVeryTight, 6=PFIsoVeryVeryTight)"),
        tkIsoId = Var("?passed('TkIsoTight')?2:passed('TkIsoLoose')","uint8",doc="TkIso ID (1=TkIsoLoose, 2=TkIsoTight)"),
        mvaId = Var("passed('MvaLoose')+passed('MvaMedium')+passed('MvaTight')+passed('MvaVTight')+passed('MvaVVTight')","uint8",doc="Mva ID from miniAOD selector (1=MvaLoose, 2=MvaMedium, 3=MvaTight, 4=MvaVTight, 5=MvaVVTight)"),
        mvaLowPtId = Var("passed('LowPtMvaLoose')+passed('LowPtMvaMedium')","uint8", doc="Low Pt Mva ID from miniAOD selector (1=LowPtMvaLoose, 2=LowPtMvaMedium)"),
        miniIsoId = Var("passed('MiniIsoLoose')+passed('MiniIsoMedium')+passed('MiniIsoTight')+passed('MiniIsoVeryTight')","uint8",doc="MiniIso ID from miniAOD selector (1=MiniIsoLoose, 2=MiniIsoMedium, 3=MiniIsoTight, 4=MiniIsoVeryTight)"),
        multiIsoId = Var("?passed('MultiIsoMedium')?2:passed('MultiIsoLoose')","uint8",doc="MultiIsoId from miniAOD selector (1=MultiIsoLoose, 2=MultiIsoMedium)"),
        puppiIsoId = Var("passed('PuppiIsoLoose')+passed('PuppiIsoMedium')+passed('PuppiIsoTight')", "uint8", doc="PuppiIsoId from miniAOD selector (1=Loose, 2=Medium, 3=Tight)"),
        triggerIdLoose = Var("passed('TriggerIdLoose')",bool,doc="TriggerIdLoose ID"), 
        inTimeMuon = Var("passed('InTimeMuon')",bool,doc="inTimeMuon ID"),
        jetNDauCharged = Var("?userCand('jetForLepJetVar').isNonnull()?userFloat('jetNDauChargedMVASel'):0", "uint8", doc="number of charged daughters of the closest jet"),
        ),
    externalVariables = cms.PSet(
        mvaTTH = ExtVar(cms.InputTag("muonMVATTH"),float, doc="TTH MVA lepton ID score",precision=14),
        mvaLowPt = ExtVar(cms.InputTag("muonMVALowPt"),float, doc="Low pt muon ID score",precision=14),
        fsrPhotonIdx = ExtVar(cms.InputTag("muonFSRassociation:fsrIndex"),int, doc="Index of the associated FSR photon"),
    ),
)


for modifier in  run2_miniAOD_80XLegacy, run2_nanoAOD_94X2016, run2_nanoAOD_94XMiniAODv1, run2_nanoAOD_94XMiniAODv2:
    modifier.toModify(muonTable.variables, puppiIsoId = None, softMva = None)

run2_nanoAOD_102Xv1.toModify(muonTable.variables, puppiIsoId = None)


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

muonSequence = cms.Sequence(slimmedMuonsUpdated+isoForMu + ptRatioRelForMu + slimmedMuonsWithUserData + finalMuons + finalLooseMuons )
muonMC = cms.Sequence(muonsMCMatchForTable + muonMCTable)
muonTables = cms.Sequence(muonFSRphotons + muonFSRassociation + muonMVATTH + muonMVALowPt + muonTable + fsrTable)

