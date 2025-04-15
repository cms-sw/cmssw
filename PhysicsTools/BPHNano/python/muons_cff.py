import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.simplePATMuonFlatTableProducer_cfi import simplePATMuonFlatTableProducer

Path=["HLT_DoubleMu4_LowMass_Displaced", "HLT_DoubleMu4_3_LowMass"]

muonBPH = cms.EDProducer("MuonTriggerSelector",
                         muonCollection = cms.InputTag("slimmedMuons"), #same collection as in NanoAOD                                                           
                         bits           = cms.InputTag("TriggerResults", "", "HLT"),
                         prescales      = cms.InputTag("patTrigger"),
                         objects        = cms.InputTag("slimmedPatTrigger"),
                         maxdR_matching = cms.double(0.3), ##for the output trigger matched collection
                         muonSelection  = cms.string("pt > 2 && abs(eta) < 2.4"), ## on the fly selection
                         HLTPaths       = cms.vstring(Path), ### comma to the softMuonsOnly
                        )

#cuts minimun number in B both mu and e, min number of trg, dz muon, dz and dr track, 
countTrgMuons = cms.EDFilter("PATCandViewCountFilter",
    minNumber = cms.uint32(2),
    maxNumber = cms.uint32(999999),
    src       = cms.InputTag("muonBPH", "SelectedMuons")
)


#muonBPHTable = cms.EDProducer("SimpleCompositeCandidateFlatTableProducer",#SimplePATMuonFlatTableProducer",
muonBPHTable = simplePATMuonFlatTableProducer.clone(
    src  = cms.InputTag("muonBPH:SelectedMuons"),
    cut  = cms.string(""), #we should not filter on cross linked collections
    name = cms.string("BPHMuon"),
    doc  = cms.string("slimmedMuons after basic selection"),
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
        dz      = Var("dB('PVDZ')", float, doc="dz (with sign) wrt first PV [cm]"),
        dzErr   = Var("abs(edB('PVDZ'))", float, doc="dz uncertainty [cm]"),
        dxy     = Var("dB('PV2D')", float, doc="dxy (with sign) wrt first PV [cm]"),
        dxyErr  = Var("edB('PV2D')", float, doc="dxy uncertainty [cm]"),
        vx      = Var("vx()", float, doc="x coordinate of vertex position [cm]"),
        vy      = Var("vy()", float, doc="y coordinate of vertex position [cm]"),
        vz      = Var("vz()", float, doc="z coordinate of vertex position [cm]"),
        ip3d    = Var("abs(dB('PV3D'))", float, doc="3D impact parameter wrt first PV [cm]"),
        sip3d   = Var("abs(dB('PV3D')/edB('PV3D'))", float, doc="3D impact parameter significance wrt first PV"),
        pfRelIso03_all = Var("(pfIsolationR03().sumChargedHadronPt + max(pfIsolationR03().sumNeutralHadronEt + pfIsolationR03().sumPhotonEt - pfIsolationR03().sumPUPt/2,0.0))/pt", float, doc="PF relative isolation dR=0.3, total (deltaBeta corrections)"),
        pfRelIso04_all = Var("(pfIsolationR04().sumChargedHadronPt + max(pfIsolationR04().sumNeutralHadronEt + pfIsolationR04().sumPhotonEt - pfIsolationR04().sumPUPt/2,0.0))/pt", float, doc="PF relative isolation dR=0.4, total (deltaBeta corrections)"),
#        isPFcand = Var("bestTrack().isPFMuon()", bool, doc="muon is PF candidate"),
        isPFcand = Var("isPFMuon",bool,doc="muon is PF candidate"),
        isGlobal    = Var("isGlobalMuon", bool, doc="muon is global muon"),
        isTracker   = Var("isTrackerMuon", bool, doc="muon is tracker muon"),
        looseId     = Var("passed('CutBasedIdLoose')", bool, doc="cut-based ID, medium WP"),
        mediumId    = Var("passed('CutBasedIdMedium')", bool, doc="cut-based ID, medium WP"),
        tightId     = Var("passed('CutBasedIdTight')", bool, doc="cut-based ID, tight WP"),
        softId      = Var("passed('SoftCutBasedId')", bool, doc="soft cut-based ID"),
        softMvaId   = Var("passed('SoftMvaId')", bool, doc="soft MVA ID"),
        pfIsoId     = Var("passed('PFIsoVeryLoose')+passed('PFIsoLoose')+passed('PFIsoMedium')+passed('PFIsoTight')+passed('PFIsoVeryTight')+passed('PFIsoVeryVeryTight')", "uint8", doc="PFIso ID from miniAOD selector (1=PFIsoVeryLoose, 2=PFIsoLoose, 3=PFIsoMedium, 4=PFIsoTight, 5=PFIsoVeryTight, 6=PFIsoVeryVeryTight)"),
        tkIsoId     = Var("?passed('TkIsoTight')?2:passed('TkIsoLoose')", "uint8", doc="TkIso ID (1=TkIsoLoose, 2=TkIsoTight)"),
        miniIsoId   = Var("passed('MiniIsoLoose')+passed('MiniIsoMedium')+passed('MiniIsoTight')+passed('MiniIsoVeryTight')", "uint8", doc="MiniIso ID from miniAOD selector (1=MiniIsoLoose, 2=MiniIsoMedium, 3=MiniIsoTight, 4=MiniIsoVeryTight)"),
        triggerIdLoose  = Var("passed('TriggerIdLoose')", bool, doc="TriggerIdLoose ID"),
        isTriggering    = Var("userInt('isTriggering')", int, doc="flag the reco muon is also triggering"),
        matched_dr      = Var("userFloat('trgDR')", float, doc="dr with the matched triggering muon"),
        matched_dpt     = Var("userFloat('trgDPT')", float, doc="dpt/pt with the matched triggering muon"),        #comma
        fired_HLT_DoubleMu4_3_LowMass = Var("userInt('HLT_DoubleMu4_3_LowMass')", int, doc="reco muon fired this trigger"),
 #       fired_HLT_DoubleMu4_LowMass_Displaced = Var("userInt('HLT_DoubleMu4_LowMass_Displaced')", int, doc="reco muon fired this trigger")
    ),
)

muonBPHMCMatch = cms.EDProducer("MCMatcher",                  # cut on deltaR, deltaPt/Pt; pick best by deltaR
    src         = muonBPHTable.src,                           # final reco collection
    matched     = cms.InputTag("finalGenParticlesBPH"),       # final mc-truth particle collection
    mcPdgId     = cms.vint32(13),                             # one or more PDG ID (13 = mu); absolute values (see below)
    checkCharge = cms.bool(False),                            # True = require RECO and MC objects to have the same charge
    mcStatus    = cms.vint32(1),                              # PYTHIA status code (1 = stable, 2 = shower, 3 = hard scattering)
    maxDeltaR   = cms.double(0.03),                           # Minimum deltaR for the match
    maxDPtRel   = cms.double(0.5),                            # Minimum deltaPt/Pt for the match
    resolveAmbiguities    = cms.bool(True),                   # Forbid two RECO objects to match to the same GEN object
    resolveByMatchQuality = cms.bool(True),                   # False = just match input in order; True = pick lowest deltaR pair first
)

muonBPHMCTable = cms.EDProducer("CandMCMatchTableProducerBPH",
    recoObjects = muonBPHTable.src,
    genParts    = cms.InputTag("finalGenParticlesBPH"),
    mcMap       = cms.InputTag("muonBPHMCMatch"),
    objName     = muonBPHTable.name,
    objType     = cms.string("Muon"), 
    objBranchName = cms.string("genPart"),
    genBranchName = cms.string("muon"),
    docString   = cms.string("MC matching to status==1 muons"),
)

allMuonTable = muonBPHTable.clone(
    src  = cms.InputTag("muonBPH:AllMuons"),
    name = cms.string("AllMuon"),
    doc  = cms.string("HLT Muons matched with reco muons"), #reco muon matched to triggering muon"),
    variables = cms.PSet(
        CandVars,
        vx = Var("vx()", float, doc="x coordinate of vertex position [cm]"),
        vy = Var("vy()", float, doc="y coordinate of vertex position [cm]"),
        vz = Var("vz()", float, doc="z coordinate of vertex position [cm]")
   )
)

muonBPHSequence   = cms.Sequence(muonBPH)
muonBPHSequenceMC = cms.Sequence(muonBPH + muonBPHMCMatch)
muonBPHTables     = cms.Sequence(muonBPHTable)
muonBPHTablesMC   = cms.Sequence(muonBPHTable + muonBPHMCTable)
