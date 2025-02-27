import FWCore.ParameterSet.Config as cms
from PhysicsTools.BPHNano.common_cff import *

########################### B-> K* ll ###########################

BToTrkTrkMuMu = cms.EDProducer(
    'BToTrkTrkLLBuilder',
    dileptons = cms.InputTag("MuMu:SelectedDiLeptons"),
    leptonTransientTracks = cms.InputTag('muonBPH', 'SelectedTransientMuons'),
    ditracks = cms.InputTag('DiTrack'),
    transientTracks = cms.InputTag('tracksBPH', 'SelectedTransientTracks'),
    PUtracks = cms.InputTag('tracksBPH', 'SelectedTracks'),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    preVtxSelection = cms.string('userFloat("min_dr") > 0.03 && ((4.5<userFloat("unfitted_B_mass_KK") && userFloat("unfitted_B_mass_KK")<6.0 ) || (4.5<userFloat("unfitted_B_mass_Kpi") && userFloat("unfitted_B_mass_Kpi")<6.0 ) || (4.5<userFloat("unfitted_B_mass_piK") && userFloat("unfitted_B_mass_piK")<6.0))'),
    postVtxSelection = cms.string('userFloat("sv_prob") > 1.e-3 && userFloat("fitted_cos_theta_2D") >= 0.90 && ( (4.6<userFloat("fitted_mass_KK") && userFloat("fitted_mass_KK")<5.8) || (4.6<userFloat("fitted_mass_Kpi") && userFloat("fitted_mass_Kpi")<5.8 ) || (4.6<userFloat("fitted_mass_piK") && userFloat("fitted_mass_piK")<5.8)) && -0.045<userFloat("trk1_svip2d") && userFloat("trk1_svip2d")<0.045 && -0.045<userFloat("trk2_svip2d") && userFloat("trk2_svip2d")<0.045'),
    dileptonMassContraint = cms.double(-1)
)

########################### Tables ###########################

BToTrkTrkMuMuTable = cms.EDProducer(
    'SimpleCompositeCandidateFlatTableProducer',
    src       = cms.InputTag("BToTrkTrkMuMu"),
    cut       = cms.string(""),
    name      = cms.string("BToTrkTrkMuMu"),
    doc       = cms.string("BToTrkTrkMuMu Variables"),
    singleton = cms.bool(False),
    extension = cms.bool(False),
    variables = cms.PSet(
        # pre-fit quantities
        CandVars,
        l1_idx      = uint('l1_idx'),
        l2_idx      = uint('l2_idx'),
        trk1_idx    = uint('trk1_idx'),
        trk2_idx    = uint('trk2_idx'),
        ditrack_idx   = uint('ditrack_idx'),
        fit_mass_KK = ufloat('fitted_mass_KK'),
        fit_mass_Kpi = ufloat('fitted_mass_Kpi'),
        fit_mass_piK = ufloat('fitted_mass_piK'),
        fit_massErr_KK = ufloat('fitted_massErr_KK'),
        fit_massErr_Kpi = ufloat('fitted_massErr_Kpi'),
        fit_massErr_piK = ufloat('fitted_massErr_piK'),
        fit_ditrack_mass_KK = ufloat('fitted_ditrack_mass_KK'),
        fit_ditrack_mass_Kpi = ufloat('fitted_ditrack_mass_Kpi'),
        fit_ditrack_mass_piK = ufloat('fitted_ditrack_mass_piK'),       
        min_dr      = ufloat('min_dr'),
        max_dr      = ufloat('max_dr'),
        # vtx info
        chi2      = ufloat('sv_chi2'),
        svprob    = ufloat('sv_prob'),
        cos2D     = ufloat('cos_theta_2D'),
        fit_cos2D = ufloat('fitted_cos_theta_2D'),
        l_xy      = ufloat('l_xy'),
        l_xy_unc  = ufloat('l_xy_unc'),
        # post-fit momentum /masses
        mll_fullfit    = ufloat('fitted_mll'),
        fit_pt         = ufloat('fitted_pt'),
        fit_eta        = ufloat('fitted_eta'),
        fit_phi        = ufloat('fitted_phi'),
        # vertex
        vtx_x   = ufloat('vtx_x'),
        vtx_y   = ufloat('vtx_y'),
        vtx_z   = ufloat('vtx_z'),
        vtx_cxx = ufloat('vtx_cxx'),
        vtx_cyy = ufloat('vtx_cyy'),
        vtx_czz = ufloat('vtx_czz'),
        vtx_cyx = ufloat('vtx_cyx'),
        vtx_czx = ufloat('vtx_czx'),
        vtx_czy = ufloat('vtx_czy'),
        # post-fit tracks/leptons
        #l1
        fit_l1_pt  = ufloat('fitted_l1_pt'),
        fit_l1_eta = ufloat('fitted_l1_eta'),
        fit_l1_phi = ufloat('fitted_l1_phi'),
        #l2
        fit_l2_pt  = ufloat('fitted_l2_pt'),
        fit_l2_eta = ufloat('fitted_l2_eta'),
        fit_l2_phi = ufloat('fitted_l2_phi'),
        #trk1
        fit_trk1_pt  = ufloat('fitted_trk1_pt'),
        fit_trk1_eta = ufloat('fitted_trk1_eta'),
        fit_trk1_phi = ufloat('fitted_trk1_phi'),
        #trk2
        fit_trk2_pt  = ufloat('fitted_trk2_pt'),
        fit_trk2_eta = ufloat('fitted_trk2_eta'),
        fit_trk2_phi = ufloat('fitted_trk2_phi'),
        # isolation 
        l1_iso04   = ufloat('l1_iso04'),
        l2_iso04   = ufloat('l2_iso04'),
        trk1_iso04 = ufloat('trk1_iso04'),
        trk2_iso04 = ufloat('trk2_iso04'),

        trk1_svip2d     = ufloat('trk1_svip2d'),
        trk1_svip2d_err = ufloat('trk1_svip2d_err'),
        trk2_svip2d     = ufloat('trk2_svip2d'),
        trk2_svip2d_err = ufloat('trk2_svip2d_err'),
    )
)

CountBToTrkTrkMuMu = cms.EDFilter("PATCandViewCountFilter",
    minNumber = cms.uint32(1),
    maxNumber = cms.uint32(999999),
    src       = cms.InputTag("BToTrkTrkMuMu")
)

########################### Sequencies  ############################
BToTrkTrkMuMuSequence = cms.Sequence( BToTrkTrkMuMu  )
BToTrkTrkMuMuTables   = cms.Sequence( BToTrkTrkMuMuTable )
