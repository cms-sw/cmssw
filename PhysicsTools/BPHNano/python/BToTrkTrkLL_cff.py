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
    dileptonMassContraint = cms.bool(True)
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
        l1_idx      = Var("userInt('l1_idx')", int, doc = "leading muon index to the BPH muon collection"),
        l2_idx      = Var("userInt('l2_idx')", int, doc = "subleading muon index to the BPH muon collection"),
        trk1_idx    = Var("userInt('trk1_idx')", int, doc = "leading track index to the BPH track collection"),
        trk2_idx    = Var("userInt('trk2_idx')", int, doc = "subleading track index to the BPH track collection"),
        ditrack_idx = Var("userInt('ditrack_idx')", int, doc = "Ditrack pair index to the BPH DiTrack collection"),
        fit_mass_KK = Var("userFloat('fitted_mass_KK')", float, doc = "mass of the B candidate for the leading trk->Kaon subleading trk->kaon mass hypothesis", precision=10),
        fit_mass_Kpi = Var("userFloat('fitted_mass_Kpi')", float, doc = "mass of the B candidate for the leading trk->Kaon subleading trk->pion mass hypothesis", precision=10),
        fit_mass_piK = Var("userFloat('fitted_mass_piK')", float, doc = "mass of the B candidate for the leading trk->pion subleading trk->kaon mass hypothesis", precision=10),
        fit_massErr_KK = Var("userFloat('fitted_massErr_KK')", float, doc = "uncertainty of the mass of the B candidate for the leading trk->Kaon subleading trk->pion mass hypothesis", precision=10),
        fit_massErr_Kpi = Var("userFloat('fitted_massErr_Kpi')", float, doc = "uncertainty of the mass of the B candidate for the leading trk->Kaon subleading trk->pion mass hypothesis", precision=10),
        fit_massErr_piK = Var("userFloat('fitted_massErr_piK')", float, doc = "uncertainty mass of the B candidate for the leading trk->Kaon subleading trk->pion mass hypothesis", precision=10),
        fit_ditrack_mass_KK = Var("userFloat('fitted_ditrack_mass_KK')", float, doc = "mass of the two track system for the leading trk->Kaon subleading trk->kaon mass hypothesis", precision=10),
        fit_ditrack_mass_Kpi = Var("userFloat('fitted_ditrack_mass_Kpi')", float, doc = "mass of the two track system for the leading trk->Kaon subleading trk->pion mass hypothesis", precision=10),
        fit_ditrack_mass_piK = Var("userFloat('fitted_ditrack_mass_piK')", float, doc = "mass of the two track system for the leading trk->pion subleading trk->kaon mass hypothesis", precision=10),
        min_dr      = Var("userFloat('min_dr')", float, doc = "min DR of the two tracks with the two muons", precision=10),
        max_dr      = Var("userFloat('max_dr')", float, doc = "max DR of the two tracks with the two muons", precision=10),
        # vtx info
        chi2      = Var("userFloat('sv_chi2')", float, doc = "vertex chi^2 of the B candidate", precision=10),
        svprob    = Var("userFloat('sv_prob')", float, doc = "vertex probability of the B candidate", precision=10),
        cos2D     = Var("userFloat('cos_theta_2D')", float, doc = "cos 2D of pre-fit candidate wrt beamspot", precision=10),
        fit_cos2D = Var("userFloat('fitted_cos_theta_2D')", float, doc = "cos 2D of fitted vertex wrt beamspot", precision=10),
        l_xy      = Var("userFloat('l_xy')", float, doc = "post-fit vertex displacement on transverse plane wrt beamspot", precision=10),
        l_xy_unc  = Var("userFloat('l_xy_unc')", float, doc = "post-fit uncertainty of the vertex displacement on transverse plane wrt beamspot", precision=10),
        # post-fit momentum /masses
        mll_fullfit    = Var("userFloat('fitted_mll')", float, doc = "post-fit mass of the two muons", precision=10),
        fit_pt         = Var("userFloat('fitted_pt')", float, doc = "post-fit B pT", precision=10),
        fit_eta        = Var("userFloat('fitted_eta')", float, doc = "post-fit B eta", precision=10),
        fit_phi        = Var("userFloat('fitted_phi')", float, doc = "post-fit B phi", precision=10),
        # vertex
        vtx_x   = Var("userFloat('vtx_x')", float, doc = "position x of fitted vertex", precision=10),
        vtx_y   = Var("userFloat('vtx_y')", float, doc = "position y of fitted vertex", precision=10),
        vtx_z   = Var("userFloat('vtx_z')", float, doc = "position z of fitted vertex", precision=10),
        vtx_cxx = Var("userFloat('vtx_cxx')", float, doc = "error x of fitted vertex", precision=10),
        vtx_cyy = Var("userFloat('vtx_cyy')", float, doc = "error y of fitted vertex", precision=10),
        vtx_czz = Var("userFloat('vtx_czz')", float, doc = "error z of fitted vertex", precision=10),
        vtx_cyx = Var("userFloat('vtx_cyx')", float, doc = "error yx of fitted vertex", precision=10),
        vtx_czx = Var("userFloat('vtx_czx')", float, doc = "error zx of fitted vertex", precision=10),
        vtx_czy = Var("userFloat('vtx_czy')", float, doc = "error zy of fitted vertex", precision=10),
        # post-fit tracks/leptons
        #l1
        fit_l1_pt  = Var("userFloat('fitted_l1_pt')", float, doc = "post-fit leading mu pT", precision=10),
        fit_l1_eta = Var("userFloat('fitted_l1_eta')", float, doc = "post-fit leading mu eta", precision=10),
        fit_l1_phi = Var("userFloat('fitted_l1_phi')", float, doc = "post-fit leading mu phi", precision=10),
        #l2
        fit_l2_pt  = Var("userFloat('fitted_l2_pt')", float, doc = "post-fit subleading mu pT", precision=10),
        fit_l2_eta = Var("userFloat('fitted_l2_eta')", float, doc = "post-fit subleading mu eta", precision=10),
        fit_l2_phi = Var("userFloat('fitted_l2_phi')", float, doc = "post-fit subleading mu phi", precision=10),
        #trk1
        fit_trk1_pt  = Var("userFloat('fitted_trk1_pt')", float, doc = "post-fit leading track pT", precision=10),
        fit_trk1_eta = Var("userFloat('fitted_trk1_eta')", float, doc = "post-fit leading track eta", precision=10),
        fit_trk1_phi = Var("userFloat('fitted_trk1_phi')", float, doc = "post-fit leading track phi", precision=10),
        #trk2
        fit_trk2_pt  = Var("userFloat('fitted_trk2_pt')", float, doc = "post-fit subleading track pT", precision=10),
        fit_trk2_eta = Var("userFloat('fitted_trk2_eta')", float, doc = "post-fit subleading track eta", precision=10),
        fit_trk2_phi = Var("userFloat('fitted_trk2_phi')", float, doc = "post-fit subleading track phi", precision=10),
        # isolation 
        l1_iso04   = Var("userFloat('l1_iso04')", float, doc = "leading mu isolation DR<0.4", precision=10),
        l2_iso04   = Var("userFloat('l2_iso04')", float, doc = "subleading mu isolation DR<0.4", precision=10),
        trk1_iso04 = Var("userFloat('trk1_iso04')", float, doc = "leading track isolation DR<0.4", precision=10),
        trk2_iso04 = Var("userFloat('trk2_iso04')", float, doc = "subleading track isolation DR<0.4", precision=10),
        trk1_svip2d     = Var("userFloat('trk1_svip2d')", float, doc = "2D IP of the leading track wrt the dimuon vertex", precision=10),
        trk1_svip2d_err = Var("userFloat('trk1_svip2d_err')", float, doc = "uncertainty of 2D IP of the leading track wrt the dimuon vertex", precision=10),
        trk2_svip2d     = Var("userFloat('trk2_svip2d')", float, doc = "2D IP of the subleading track wrt the dimuon vertex", precision=10),
        trk2_svip2d_err = Var("userFloat('trk2_svip2d_err')", float, doc = "uncertainty of 2D IP of the subleading track wrt the dimuon vertex", precision=10),
        constraint_sv_prob     = Var("userFloat('constraint_sv_prob')", float, doc = "B vertex probability after the dimuon mass constraint", precision=10),
        constraint_pt   = Var("userFloat('constraint_pt')", float, doc = "B pt after the dimuon mass constraint", precision=10),
        constraint_eta  = Var("userFloat('constraint_eta')", float, doc = "B eta after the dimuon mass constraint", precision=10),
        constraint_phi  = Var("userFloat('constraint_phi')", float, doc = "B phi after the dimuon mass constraint", precision=10),
        constraint_mass_KK = Var("userFloat('constraint_mass_KK')", float, doc = "B mass with KK mass hypothesis after the dimuon mass constraint", precision=10),
        constraint_massErr_KK  = Var("userFloat('constraint_massErr_KK')", float, doc = "mass uncertainty for the KK mass hypothesis of the dimuon mass constraint", precision=10),
        constraint_mass_Kpi = Var("userFloat('constraint_mass_Kpi')", float, doc = "B mass with Kpi mass hypothesis after the dimuon mass constraint", precision=10),
        constraint_massErr_Kpi  = Var("userFloat('constraint_massErr_Kpi')", float, doc = "mass uncertainty for the Kpi mass hypothesis of the dimuon mass constraint", precision=10),
        constraint_mass_piK = Var("userFloat('constraint_mass_piK')", float, doc = "B mass with piK mass hypothesis after the dimuon mass constraint", precision=10),
        constraint_massErr_piK  = Var("userFloat('constraint_massErr_piK')", float, doc = "mass uncertainty for the piK mass hypothesis of the dimuon mass constraint", precision=10),
        constraint_mll  = Var("userFloat('constraint_mll')", float, doc = "dimuon mass after the dimuon mass constraint", precision=10),
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
