import FWCore.ParameterSet.Config as cms
from PhysicsTools.BPHNano.common_cff import *

########################### Selections ###########################

KshortToPiPi = cms.EDProducer(
    'V0ReBuilder',
    V0s = cms.InputTag('slimmedKshortVertices'),
    trkSelection = cms.string('pt > 0.35 && abs(eta) < 2.5 && trackHighPurity()'),
    V0Selection = cms.string('0.3 < mass && mass < 0.7'),
    postVtxSelection = cms.string('0.3 < mass && mass < 0.7'
                                  '&& userFloat("sv_prob") > 0.0001'),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    track_match = cms.InputTag('tracksBPH', 'SelectedTracks')
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
        chi2 = ufloat('sv_chi2'),
        svprob = ufloat('sv_prob'),
        l_xy = ufloat('l_xy'),
        l_xy_unc = ufloat('l_xy_unc'),
        prefit_mass = ufloat('prefit_mass'),
        vtx_x = ufloat('vtx_x'),
        vtx_y = ufloat('vtx_y'),
        vtx_z = ufloat('vtx_z'),
        vtx_cxx = ufloat('vtx_cxx'),
        vtx_cyy = ufloat('vtx_cyy'),
        vtx_czz = ufloat('vtx_czz'),
        vtx_cyx = ufloat('vtx_cyx'),
        vtx_czx = ufloat('vtx_czx'),
        vtx_czy = ufloat('vtx_czy'),
        fit_cos_theta_2D = ufloat('fitted_cos_theta_2D'),        
        # post-fit momentum
        fit_massErr = ufloat('massErr'),        
        fit_trk1_pt = ufloat('trk1_pt'),
        fit_trk1_eta = ufloat('trk1_eta'),
        fit_trk1_phi = ufloat('trk1_phi'),
        fit_trk2_pt = ufloat('trk2_pt'),
        fit_trk2_eta = ufloat('trk2_eta'),
        fit_trk2_phi = ufloat('trk2_phi'),
        # track match
        trk1_idx = uint('trk1_idx'),
        trk2_idx = uint('trk2_idx'),
    )
)

CountKshortToPiPi = cms.EDFilter("PATCandViewCountFilter",
    minNumber = cms.uint32(1),
    maxNumber = cms.uint32(999999),
    src = cms.InputTag('KshortToPiPi','SelectedV0Collection')
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

KshortPiPiBPHMCTable = cms.EDProducer("CandMCMatchTableProducerBPH",
    recoObjects = KshortToPiPiTable.src,
    genParts = cms.InputTag("finalGenParticlesBPH"),
    mcMap = cms.InputTag("KshortPiPiBPHMCMatch"),
    objName = KshortToPiPiTable.name,
    objType = cms.string("Other"),
    objBranchName = cms.string("genPart"),
    genBranchName = cms.string("kshort"),
    docString = cms.string("MC matching to status==1 muons"),
)

KshortToPiPiSequence = cms.Sequence( KshortToPiPi )
KshortToPiPiSequenceMC = cms.Sequence( KshortToPiPi +KshortPiPiBPHMCMatch)
KshortToPiPiTables = cms.Sequence( KshortToPiPiTable)
KshortToPiPiTablesMC = cms.Sequence( KshortToPiPiTable+KshortPiPiBPHMCTable)
