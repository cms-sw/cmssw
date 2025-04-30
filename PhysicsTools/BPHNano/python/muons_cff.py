import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.simplePATMuonFlatTableProducer_cfi import simplePATMuonFlatTableProducer

Path=["HLT_DoubleMu4_3_LowMass"]

# Takes slimmedMuons, apply basic preselection and trigger match. Devides muon in
# SelectedMuons (trigger matched) and AllMuons collections
muonBPH = cms.EDProducer("MuonTriggerSelector",
    muonCollection = cms.InputTag("slimmedMuons"),                                                        
    bits           = cms.InputTag("TriggerResults", "", "HLT"),
    prescales      = cms.InputTag("patTrigger"),
    objects        = cms.InputTag("slimmedPatTrigger"),
    maxdR_matching = cms.double(0.3), # For the output trigger matched collection
    muonSelection  = cms.string("pt > 2 && abs(eta) < 2.4"), # Preselection
    HLTPaths       = cms.vstring(Path)
)

# Cuts minimun number in B both mu
countTrgMuons = cms.EDFilter("PATCandViewCountFilter",
    minNumber = cms.uint32(2),
    maxNumber = cms.uint32(999999),
    src       = cms.InputTag("muonBPH", "SelectedMuons")
)

# Table containing triggering muons, to interface with signal reconstruction
# Contains variable mostly used for signal reconstruction and analysis
TrgMatchMuonTable = simplePATMuonFlatTableProducer.clone(
    src  = cms.InputTag("muonBPH:SelectedMuons"), # Could this be removed and simply select userInt('isTriggering')==1
    cut  = cms.string(""), # We should not filter on cross linked collections
    name = cms.string("TrgMatchMuon"),
    doc  = cms.string("slimmedMuons after basic selection and trigger match"),
    singleton = cms.bool(False), # the number of entries is variable
    extension = cms.bool(False), # this is the main table for the muons
    variables = cms.PSet(
        CandVars,
        ptErr   = Var("bestTrack().ptError()", float, doc="ptError of the muon track"),
        covQopQop = Var("bestTrack().covariance(0, 0)", float, doc="Cov of q/p with q/p", precision=10),
        covLamLam = Var("bestTrack().covariance(1, 1)", float, doc="Cov of lambda with lambda", precision=10),
        covPhiPhi = Var("bestTrack().covariance(2, 2)", float, doc="Cov of phi with phi", precision=10),
        covQopLam = Var("bestTrack().covariance(0, 1)", float, doc="Cov of q/p with lambda", precision=10),
        covQopPhi = Var("bestTrack().covariance(0, 2)", float, doc="Cov of q/p with phi", precision=10),
        covLamPhi = Var("bestTrack().covariance(1, 2)", float, doc="Cov of lambda with phi", precision=10),
        dz      = Var("dB('PVDZ')", float, doc="dz (with sign) wrt PV[0] [cm]"),
        dzErr   = Var("abs(edB('PVDZ'))", float, doc="dz uncertainty [cm]"),
        dxy     = Var("dB('PV2D')", float, doc="dxy (with sign) wrt PV[0] [cm]"),
        dxyErr  = Var("edB('PV2D')", float, doc="dxy uncertainty [cm]"),
        ip3d    = Var("abs(dB('PV3D'))", float, doc="3D impact parameter wrt PV[0] [cm]"),
        sip3d   = Var("abs(dB('PV3D')/edB('PV3D'))", float, doc="3D impact parameter significance wrt PV[0]"),
        pfRelIso03_all = Var("(pfIsolationR03().sumChargedHadronPt + max(pfIsolationR03().sumNeutralHadronEt + pfIsolationR03().sumPhotonEt - pfIsolationR03().sumPUPt/2,0.0))/pt", float, doc="PF relative isolation dR=0.3, total (deltaBeta corrections)"),
        pfRelIso04_all = Var("(pfIsolationR04().sumChargedHadronPt + max(pfIsolationR04().sumNeutralHadronEt + pfIsolationR04().sumPhotonEt - pfIsolationR04().sumPUPt/2,0.0))/pt", float, doc="PF relative isolation dR=0.4, total (deltaBeta corrections)"),
        isPFcand    = Var("isPFMuon", bool, doc="muon is PF candidate"),
        isGlobal    = Var("isGlobalMuon", bool, doc="muon is global muon"),
        isTracker   = Var("isTrackerMuon", bool, doc="muon is tracker muon"),
        looseId     = Var("passed('CutBasedIdLoose')", bool, doc="cut-based ID, medium WP"),
        mediumId    = Var("passed('CutBasedIdMedium')", bool, doc="cut-based ID, medium WP"),
        # tightId     = Var("passed('CutBasedIdTight')", bool, doc="cut-based ID, tight WP"),
        triggerIdLoose  = Var("passed('TriggerIdLoose')", bool, doc="TriggerIdLoose ID"),
        softId = Var("passed('SoftCutBasedId')",bool,doc="soft cut-based ID"),
        softMvaId = Var("passed('SoftMvaId')",bool,doc="soft MVA ID"),
        softMva = Var("softMvaValue()",float,doc="soft MVA ID score",precision=6),
        softMvaRun3 = Var("softMvaRun3Value()",float,doc="soft MVA Run3 ID score",precision=6),
        isTriggering    = Var("userInt('isTriggering')", int, doc="flag the reco muon if matched to HLT object"),
        matched_dr      = Var("userFloat('trgDR')", float, doc="dr with the matched triggering muon"),
        matched_dpt     = Var("userFloat('trgDPT')", float, doc="dpt/pt with the matched triggering muon"), #comma
        # fired_HLT_DoubleMu4_3_LowMass = Var("userInt('HLT_DoubleMu4_3_LowMass')", int, doc="reco muon fired this trigger"),
        # fired_HLT_DoubleMu4_LowMass_Displaced = Var("userInt('HLT_DoubleMu4_LowMass_Displaced')", int, doc="reco muon fired this trigger")
    ),
)

# Producer for MC matching
muonBPHMCMatch = cms.EDProducer("MCMatcher",                  # cut on deltaR, deltaPt/Pt; pick best by deltaR
    src         = TrgMatchMuonTable.src,                      # final reco collection
    matched     = cms.InputTag("finalGenParticlesBPH"),       # final mc-truth particle collection
    mcPdgId     = cms.vint32(13),                             # one or more PDG ID (13 = mu); absolute values (see below)
    checkCharge = cms.bool(False),                            # True = require RECO and MC objects to have the same charge
    mcStatus    = cms.vint32(1),                              # PYTHIA status code (1 = stable, 2 = shower, 3 = hard scattering)
    maxDeltaR   = cms.double(0.05),                           # Minimum deltaR for the match
    maxDPtRel   = cms.double(0.5),                            # Minimum deltaPt/Pt for the match
    resolveAmbiguities    = cms.bool(True),                   # Forbid two RECO objects to match to the same GEN object
    resolveByMatchQuality = cms.bool(True),                   # False = just match input in order; True = pick lowest deltaR pair first
)

# Table for MC matching
MCMuonTable = cms.EDProducer("CandMCMatchTableProducer",
    src = TrgMatchMuonTable.src,
    mcMap       = cms.InputTag("muonBPHMCMatch"),
    objName     = TrgMatchMuonTable.name,
    objType     = cms.string("Muon"),
    branchName = cms.string("genPart"),
    docString   = cms.string("MC matching to status==1 muons"),
)


muonBPHSequence   = cms.Sequence(muonBPH)
muonBPHSequenceMC = cms.Sequence(muonBPH + muonBPHMCMatch)
muonBPHTables   = cms.Sequence(TrgMatchMuonTable)
muonBPHTablesMC = cms.Sequence(TrgMatchMuonTable + MCMuonTable)

