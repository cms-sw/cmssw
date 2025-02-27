import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *

tracksBPH = cms.EDProducer(
    "TrackMerger",
    beamSpot        = cms.InputTag("offlineBeamSpot"),
    dileptons       = cms.InputTag("MuMu:SelectedDiLeptons"),
    tracks          = cms.InputTag("packedPFCandidates"),
    lostTracks      = cms.InputTag("lostTracks"),
    trackSelection  = cms.string("pt>0.7 && abs(eta)<2.4"),  # We need all tracks for tagging, no cuts here for now
    muons           = cms.InputTag("slimmedMuons"),
    electrons       = cms.InputTag("slimmedElectrons"),
    maxDzDilep      = cms.double(1.0),
    dcaSig          = cms.double(-100000),
)


trackBPHTable = cms.EDProducer(
    "SimpleCompositeCandidateFlatTableProducer",
    src  = cms.InputTag("tracksBPH:SelectedTracks"),
    cut  = cms.string(""),
    name = cms.string("Track"),
    doc  = cms.string("track collection"),
    singleton = cms.bool(False),
    extension = cms.bool(False), 
    variables = cms.PSet(
        CandVars,
        vx = Var("vx()", float, doc="x coordinate of vtx position [cm]"),
        vy = Var("vy()", float, doc="y coordinate of vtx position [cm]"),
        vz = Var("vz()", float, doc="z coordinate of vtx position [cm]"),
        # User variables defined in plugins/TrackMerger.cc
        isPacked  = Var("userInt('isPacked')", bool, doc="track from packedCandidate collection"),
        isLostTrk = Var("userInt('isLostTrk')", bool, doc="track from lostTrack collection"),
        dz      = Var("userFloat('dz')", float, doc="dz signed wrt first PV [cm]"),
        dxy     = Var("userFloat('dxy')", float, doc="dxy (with sign) wrt first PV [cm]"),
        dzS     = Var("userFloat('dzS')", float, doc="dz/err (with sign) wrt first PV [cm]"),
        dxyS    = Var("userFloat('dxyS')", float, doc="dxy/err (with sign) wrt first PV [cm]"),
        DCASig  = Var("userFloat('DCASig')", float, doc="significance of xy-distance of closest approach wrt beamspot"),
        dzTrg   = Var("userFloat('dzTrg')", float, doc="dz from the corresponding trigger muon [cm]"),
        isMatchedToMuon = Var("userInt('isMatchedToMuon')", bool, doc="track was used to build a muon"),
        isMatchedToEle  = Var("userInt('isMatchedToEle')", bool, doc="track was used to build a PF ele"),
        nValidHits      = Var("userInt('nValidHits')", int, doc="Number of valid hits"),
        # Covariance matrix elements for helix parameters for decay time uncertainty
        covQopQop = Var("userFloat('covQopQop')", float, doc="Cov. of q/p with q/p", precision=10),
        covQopLam = Var("userFloat('covQopLam')", float, doc="Cov. of q/p with lambda", precision=10),
        covQopPhi = Var("userFloat('covQopPhi')", float, doc="Cov. of q/p with phi", precision=10),
        covLamLam = Var("userFloat('covLamLam')", float, doc="Cov. of lambda with lambda", precision=10),
        covLamPhi = Var("userFloat('covLamPhi')", float, doc="Cov. of lambda with phi", precision=10),
        covPhiPhi = Var("userFloat('covPhiPhi')", float, doc="Cov. of phi with phi", precision=10),
        # Additional track parameters for tagging
        ptErr      = Var("userFloat('ptErr')", float, doc="Pt uncertainty"),
        normChi2   = Var("userFloat('normChi2')", float, doc="Track fit chi-squared divided by n.d.o.f."),
        nValidPixelHits = Var("userInt('nValidPixelHits')", float, doc="Number of pixel hits"),
        # TODO: check impact parameters
        ),
)


tracksBPHMCMatch = cms.EDProducer("MCMatcher",              # cut on deltaR, deltaPt/Pt; pick best by deltaR
    src         = trackBPHTable.src,                        # final reco collection
    matched     = cms.InputTag("finalGenParticlesBPH"),     # final mc-truth particle collection
    mcPdgId     = cms.vint32(321, 211),                     # one or more PDG ID (321 = charged kaon, 211 = charged pion); absolute values (see below)
    checkCharge = cms.bool(False),                          # True = require RECO and MC objects to have the same charge
    mcStatus    = cms.vint32(1),                            # PYTHIA status code (1 = stable, 2 = shower, 3 = hard scattering)
    maxDeltaR   = cms.double(0.03),                         # Minimum deltaR for the match
    maxDPtRel   = cms.double(0.5),                          # Minimum deltaPt/Pt for the match
    resolveAmbiguities    = cms.bool(True),                 # Forbid two RECO objects to match to the same GEN object
    resolveByMatchQuality = cms.bool(True),                 # False = just match input in order; True = pick lowest deltaR pair first
)


tracksBPHMCTable = cms.EDProducer("CandMCMatchTableProducerBPH",
    recoObjects   = tracksBPHMCMatch.src,
    genParts      = cms.InputTag("finalGenParticlesBPH"),
    mcMap         = cms.InputTag("tracksBPHMCMatch"),
    objName       = trackBPHTable.name,
    objType       = trackBPHTable.name,
    objBranchName = cms.string("genPart"),
    genBranchName = cms.string("track"),
    docString     = cms.string("MC matching to status==1 kaons or pions"),
)


tracksBPHSequence   = cms.Sequence(tracksBPH)
tracksBPHSequenceMC = cms.Sequence(tracksBPH + tracksBPHMCMatch)
tracksBPHTables     = cms.Sequence(trackBPHTable)
tracksBPHTablesMC   = cms.Sequence(trackBPHTable + tracksBPHMCTable)
