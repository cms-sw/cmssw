import FWCore.ParameterSet.Config as cms
from PhysicsTools.BPHNano.common_cff import *

########################### B-> K* ll ##########################

BToKshortMuMu = cms.EDProducer(
    'BToV0LLBuilder',
    dileptons = cms.InputTag("MuMu:SelectedDiLeptons"),
    leptonTransientTracks = cms.InputTag('muonBPH', 'SelectedTransientMuons'),
    v0s = cms.InputTag('KshortToPiPi','SelectedV0Collection'),
    v0TransientTracks = cms.InputTag('KshortToPiPi', 'SelectedV0TransientCollection'),
    tracks = cms.InputTag("packedPFCandidates"),
    PUtracks = cms.InputTag('tracksBPH', 'SelectedTracks'),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    preVtxSelection  = cms.string('pt > 5.0 '
                                  '&& 4.5 < mass && mass < 6. '
                                  '&& userFloat("min_dr") > 0.03'),
    postVtxSelection = cms.string('4.7 < userFloat("fitted_mass") && userFloat("fitted_mass") < 6. '
                                  '&& userFloat("sv_prob") > 0.001 && -0.045<userFloat("v0_svip2d") && userFloat("v0_svip2d")<0.045'
                                  '&& userFloat("fitted_cos_theta_2D") > 0.9'),
    dileptonMassContraint = cms.double(-1)
)

########################### Tables ###########################

BToKshortMuMuTable = cms.EDProducer(
    'SimpleCompositeCandidateFlatTableProducer',
    src       = cms.InputTag("BToKshortMuMu"),
    cut       = cms.string(""),
    name      = cms.string("BToKshortMuMu"),
    doc       = cms.string("BToKshortMuMu Variables"),
    singleton = cms.bool(False),
    extension = cms.bool(False),
    variables = cms.PSet(
        # pre-fit quantities
        CandVars,
        l1_idx      = uint('l1_idx'),
        l2_idx      = uint('l2_idx'),
        kshort_idx  = uint('v0_idx'),
        min_dr      = ufloat('min_dr'),
        max_dr      = ufloat('max_dr'),
        # fit and vtx info
        chi2      = ufloat('sv_chi2'),
        svprob    = ufloat('sv_prob'),
        l_xy      = ufloat('l_xy'),
        l_xy_unc  = ufloat('l_xy_unc'),
        cos2D     = ufloat('cos_theta_2D'),
        fit_cos2D = ufloat('fitted_cos_theta_2D'),
        vtx_x     = ufloat('vtx_x'),
        vtx_y     = ufloat('vtx_y'),
        vtx_z     = ufloat('vtx_z'),
        vtx_cxx   = ufloat('vtx_cxx'),
        vtx_cyy   = ufloat('vtx_cyy'),
        vtx_czz   = ufloat('vtx_czz'),
        vtx_cyx   = ufloat('vtx_cyx'),
        vtx_czx   = ufloat('vtx_czx'),
        vtx_czy   = ufloat('vtx_czy'),
        # post fit properties
        mll_fullfit     = ufloat('fitted_mll'),     
        mkshort_fullfit = ufloat('fitted_v0_mass'),
        fit_mass        = ufloat('fitted_mass'),
        fit_massErr     = ufloat('fitted_massErr'),
        fit_pt          = ufloat('fitted_pt'),
        fit_eta         = ufloat('fitted_eta'),
        fit_phi         = ufloat('fitted_phi'),
        # post-fit tracks/leptons
        #l1
        fit_l1_pt  = ufloat('fitted_l1_pt'),
        fit_l1_eta = ufloat('fitted_l1_eta'),
        fit_l1_phi = ufloat('fitted_l1_phi'),
        #l2
        fit_l2_pt  = ufloat('fitted_l2_pt'),
        fit_l2_eta = ufloat('fitted_l2_eta'),
        fit_l2_phi = ufloat('fitted_l2_phi'),
        #Kshort
        fit_kshort_pt  = ufloat('fitted_v0_pt'),
        fit_kshort_eta = ufloat('fitted_v0_eta'),
        fit_kshort_phi = ufloat('fitted_v0_phi'),
        # isolation and ipd2d
        kshort_svip2d     = ufloat('v0_svip2d'),
        kshort_svip2d_err = ufloat('v0_svip2d_err'),
        l1_iso04          = ufloat('l1_iso04'),
        l2_iso04          = ufloat('l2_iso04'),
        kshort_iso04      = ufloat('v0_iso04'),
    )
)

CountBToKshortMuMu = cms.EDFilter("PATCandViewCountFilter",
    minNumber = cms.uint32(1),
    maxNumber = cms.uint32(999999),
    src       = cms.InputTag("BToKshortMuMu")
)

########################### Sequencies  ############################
BToKshortMuMuSequence = cms.Sequence( BToKshortMuMu )
BToKshortMuMuTables   = cms.Sequence( BToKshortMuMuTable )
