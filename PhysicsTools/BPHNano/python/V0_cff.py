import FWCore.ParameterSet.Config as cms
from PhysicsTools.BPHNano.common_cff import *

########################### Selections ###########################

KshortToPiPi = cms.EDProducer(
    'V0ReBuilder',
    V0s = cms.InputTag('slimmedKshortVertices'),
    trkSelection = cms.string('pt > 0.35 && abs(eta) < 3.0 && trackHighPurity()'),
    V0Selection = cms.string('0.3 < mass && mass < 0.7'),
    postVtxSelection = cms.string('0.3 < mass && mass < 0.7'
                                  '&& userFloat("sv_prob") > 0.0001'),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    track_match = cms.InputTag('tracksBPH', 'SelectedTracks'),
    isLambda = cms.bool(False)
)

LambdaToProtonPi = cms.EDProducer(
    'V0ReBuilder',
    V0s = cms.InputTag('slimmedLambdaVertices'),
    trkSelection = cms.string('pt > 0.35 && abs(eta) < 3.0 && trackHighPurity()'),
    V0Selection = cms.string('1 < mass && mass < 1.2'),
    postVtxSelection = cms.string('1 < mass && mass < 1.17'
                                  '&& userFloat("sv_prob") > 0.0001'),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    track_match = cms.InputTag('tracksBPH', 'SelectedTracks'),
    isLambda = cms.bool(True)    
)


########################### Tables ###########################

KshortToPiPiTable = cms.EDProducer(
    'SimpleCompositeCandidateFlatTableProducer',
    src = cms.InputTag('KshortToPiPi','SelectedV0Collection'),
    cut = cms.string(""),
    name = cms.string("Kshort"),
    doc = cms.string("Kshort Variables"),
    singleton=cms.bool(False),
    extension=cms.bool(False),
    variables = cms.PSet(
        # pre-fit quantities    
        CandVars,
        # fit and vtx info        
        chi2 = Var("userFloat('sv_chi2')", float, doc = "chi2 of fitted vertex", precision=10),
        svprob = Var("userFloat('sv_prob')", float, doc = "vertex probability of fitted vertex", precision=10),
        l_xy = Var("userFloat('l_xy')", float, doc = "post-fit vertex displacement on transverse plane", precision=10),
        l_xy_unc = Var("userFloat('l_xy_unc')", float, doc = "post-fit vertex uncertainty of the diplacement on the transverse plane", precision=10),
        prefit_mass = Var("userFloat('prefit_mass')", float, doc = "pre-fit mass of the vertex", precision=10),
        vtx_x = Var("userFloat('vtx_x')", float, doc = "x position of fitted vertex", precision=10),
        vtx_y = Var("userFloat('vtx_y')", float, doc = "y position of fitted vertex", precision=10),
        vtx_z = Var("userFloat('vtx_z')", float, doc = "z position of fitted vertex", precision=10),
        vtx_cxx = Var("userFloat('vtx_cxx')", float, doc = "error x of fitted vertex", precision=10),
        vtx_cyy = Var("userFloat('vtx_cyy')", float, doc = "error y of fitted vertex", precision=10),
        vtx_czz = Var("userFloat('vtx_czz')", float, doc = "error z of fitted vertex", precision=10),
        vtx_cyx = Var("userFloat('vtx_cyx')", float, doc = "error yx of fitted vertex", precision=10),
        vtx_czx = Var("userFloat('vtx_czx')", float, doc = "error zx of fitted vertex", precision=10),
        vtx_czy = Var("userFloat('vtx_czy')", float, doc = "error zy of fitted vertex", precision=10),
        fit_cos_theta_2D = Var("userFloat('fitted_cos_theta_2D')", float, doc = "cos 2D of fitted vertex wrt beamspot", precision=10),
        # post-fit momentum
        fit_mass = Var("userFloat('fitted_mass')", float, doc = "post-fit mass of the vertex", precision=10),
        fit_massErr = Var("userFloat('massErr')", float, doc = "post-fit mass error", precision=10),
        fit_trk1_pt = Var("userFloat('trk1_pt')", float, doc = "post-fit pt of the leading track", precision=10),
        fit_trk1_eta = Var("userFloat('trk1_eta')", float, doc = "post-fit eta of the leading track", precision=10),
        fit_trk1_phi = Var("userFloat('trk1_phi')", float, doc = "post-fit phi of the leading track", precision=10),
        fit_trk2_pt = Var("userFloat('trk2_pt')", float, doc = "post-fit pt of the subleading track", precision=10),
        fit_trk2_eta = Var("userFloat('trk2_eta')", float, doc = "post-fit eta of the subleading track", precision=10),
        fit_trk2_phi = Var("userFloat('trk2_phi')", float, doc = "post-fit phi of the subleading track", precision=10),
        # track match
        trk1_idx = Var("userInt('trk1_idx')", int, doc = "leading track index to the BPH track collection"),
        trk2_idx = Var("userInt('trk2_idx')", int, doc = "subleading track index to the BPH track collection"),
    )
)

CountKshortToPiPi = cms.EDFilter("PATCandViewCountFilter",
    minNumber = cms.uint32(1),
    maxNumber = cms.uint32(999999),
    src = cms.InputTag('KshortToPiPi','SelectedV0Collection')
)

LambdaToProtonPiTable = KshortToPiPiTable.clone(
    src = cms.InputTag('LambdaToProtonPi','SelectedV0Collection'),
    name = cms.string("Lambda"),
    doc = cms.string("Lambda Variable")
)

CountLambdaToProtonPi = cms.EDFilter("PATCandViewCountFilter",
    minNumber = cms.uint32(1),
    maxNumber = cms.uint32(999999),
    src = cms.InputTag('LambdaToProtonPi','SelectedV0Collection')
)


KshortPiPiBPHMCMatch = cms.EDProducer("MCMatcher",            # cut on deltaR, deltaPt/Pt; pick best by deltaR
    src         = KshortToPiPiTable.src,                      # final reco collection
    matched     = cms.InputTag("finalGenParticlesBPH"),       # final mc-truth particle collection
    mcPdgId     = cms.vint32(310),                            # one or more PDG ID (13 = mu); absolute values (see below)
    checkCharge = cms.bool(False),                            # True = require RECO and MC objects to have the same charge
    mcStatus    = cms.vint32(2),                              # PYTHIA status code (1 = stable, 2 = shower, 3 = hard scattering)
    maxDeltaR   = cms.double(0.3),                            # Minimum deltaR for the match
    maxDPtRel   = cms.double(1.0),                            # Minimum deltaPt/Pt for the match
    resolveAmbiguities    = cms.bool(True),                   # Forbid two RECO objects to match to the same GEN object
    resolveByMatchQuality = cms.bool(True),                   # False = just match input in order; True = pick lowest deltaR pair first
)

LambdaProtonPiBPHMCMatch = cms.EDProducer("MCMatcher",        # cut on deltaR, deltaPt/Pt; pick best by deltaR
    src         = LambdaToProtonPiTable.src,                  # final reco collection
    matched     = cms.InputTag("finalGenParticlesBPH"),       # final mc-truth particle collection
    mcPdgId     = cms.vint32(3122),                           # one or more PDG ID (13 = mu); absolute values (see below)
    checkCharge = cms.bool(False),                            # True = require RECO and MC objects to have the same charge
    mcStatus    = cms.vint32(2),                              # PYTHIA status code (1 = stable, 2 = shower, 3 = hard scattering)
    maxDeltaR   = cms.double(0.3),                            # Minimum deltaR for the match
    maxDPtRel   = cms.double(1.0),                            # Minimum deltaPt/Pt for the match
    resolveAmbiguities    = cms.bool(True),                   # Forbid two RECO objects to match to the same GEN object
    resolveByMatchQuality = cms.bool(True),                   # False = just match input in order; True = pick lowest deltaR pair first
)


KshortPiPiBPHMCTable = cms.EDProducer("CandMCMatchTableProducer",
    src = KshortToPiPiTable.src,
    mcMap = cms.InputTag("KshortPiPiBPHMCMatch"),
    objName = KshortToPiPiTable.name,
    objType = cms.string("Other"),
    branchName = cms.string("genPart"),
    docString = cms.string("MC matching to status==1 muons"),
)


LambdaProtonPiBPHMCTable = cms.EDProducer("CandMCMatchTableProducer",
    src = LambdaToProtonPiTable.src,
    mcMap = cms.InputTag("LambdaProtonPiBPHMCMatch"),
    objName = LambdaToProtonPiTable.name,
    objType = cms.string("Other"),
    branchName = cms.string("genPart"),
    docString = cms.string("MC matching to status==1 muons"),
)

KshortToPiPiSequence = cms.Sequence( KshortToPiPi )
KshortToPiPiSequenceMC = cms.Sequence( KshortToPiPi +KshortPiPiBPHMCMatch)
KshortToPiPiTables = cms.Sequence( KshortToPiPiTable)
KshortToPiPiTablesMC = cms.Sequence( KshortToPiPiTable+KshortPiPiBPHMCTable)


LambdaToProtonPiSequence = cms.Sequence( LambdaToProtonPi )
LambdaToProtonPiSequenceMC = cms.Sequence( LambdaToProtonPi + LambdaProtonPiBPHMCMatch)
LambdaToProtonPiTables = cms.Sequence( LambdaToProtonPiTable)
LambdaToProtonPiTablesMC = cms.Sequence( LambdaToProtonPiTable+LambdaProtonPiBPHMCTable)


