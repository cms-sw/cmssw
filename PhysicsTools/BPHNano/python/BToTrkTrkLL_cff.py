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
    src       = cms.InputTag("BToTrkTrkMuMu:SelectedBToTrkTrkMuMu"),
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
        ll_idx  = Var("userInt('ll_idx')", int, doc = "dilepton index to the MuMu collection"),
        fit_mass_KK = Var("userFloat('fitted_mass_KK')", float, doc = "mass of the B candidate for the leading trk->Kaon subleading trk->kaon mass hypothesis"),
        fit_mass_Kpi = Var("userFloat('fitted_mass_Kpi')", float, doc = "mass of the B candidate for the leading trk->Kaon subleading trk->pion mass hypothesis"),
        fit_mass_piK = Var("userFloat('fitted_mass_piK')", float, doc = "mass of the B candidate for the leading trk->pion subleading trk->kaon mass hypothesis"),
        fit_massErr_KK = Var("userFloat('fitted_massErr_KK')", float, doc = "uncertainty of the mass of the B candidate for the leading trk->Kaon subleading trk->pion mass hypothesis", precision=12),
        fit_massErr_Kpi = Var("userFloat('fitted_massErr_Kpi')", float, doc = "uncertainty of the mass of the B candidate for the leading trk->Kaon subleading trk->pion mass hypothesis", precision=12),
        fit_massErr_piK = Var("userFloat('fitted_massErr_piK')", float, doc = "uncertainty mass of the B candidate for the leading trk->Kaon subleading trk->pion mass hypothesis", precision=12),
        fit_ditrack_mass_KK = Var("userFloat('fitted_ditrack_mass_KK')", float, doc = "mass of the two track system for the leading trk->Kaon subleading trk->kaon mass hypothesis"),
        fit_ditrack_mass_Kpi = Var("userFloat('fitted_ditrack_mass_Kpi')", float, doc = "mass of the two track system for the leading trk->Kaon subleading trk->pion mass hypothesis"),
        fit_ditrack_mass_piK = Var("userFloat('fitted_ditrack_mass_piK')", float, doc = "mass of the two track system for the leading trk->pion subleading trk->kaon mass hypothesis"),
        min_dr      = Var("userFloat('min_dr')", float, doc = "min DR of the two tracks with the two muons", precision=12),
        max_dr      = Var("userFloat('max_dr')", float, doc = "max DR of the two tracks with the two muons", precision=12),
        # vtx info
        chi2      = Var("userFloat('sv_chi2')", float, doc = "vertex chi^2 of the B candidate", precision=12),
        svprob    = Var("userFloat('sv_prob')", float, doc = "vertex probability of the B candidate"),
        cos2D     = Var("userFloat('cos_theta_2D')", float, doc = "cos 2D of pre-fit candidate wrt beamspot", precision=12),
        fit_cos2D = Var("userFloat('fitted_cos_theta_2D')", float, doc = "cos 2D of fitted vertex wrt beamspot"),
        l_xy      = Var("userFloat('l_xy')", float, doc = "post-fit vertex displacement on transverse plane wrt beamspot"),
        l_xy_unc  = Var("userFloat('l_xy_unc')", float, doc = "post-fit uncertainty of the vertex displacement on transverse plane wrt beamspot"),
        # post-fit momentum /masses
        mll_fullfit    = Var("userFloat('fitted_mll')", float, doc = "post-fit mass of the two muons"),
        fit_pt         = Var("userFloat('fitted_pt')", float, doc = "post-fit B pT"),
        fit_eta        = Var("userFloat('fitted_eta')", float, doc = "post-fit B eta"),
        fit_phi        = Var("userFloat('fitted_phi')", float, doc = "post-fit B phi"),
        # vertex
        dca = Var("userFloat('dca')", float, doc = "DCA of B candidate wrt to beamspot", precision=12),
        dcaErr = Var("userFloat('dcaErr')", float, doc = "Error DCA of B candidate wrt to beamspot", precision=12),
        vtx_x   = Var("userFloat('vtx_x')", float, doc = "position x of fitted vertex", precision=12),
        vtx_y   = Var("userFloat('vtx_y')", float, doc = "position y of fitted vertex", precision=12),
        vtx_z   = Var("userFloat('vtx_z')", float, doc = "position z of fitted vertex", precision=12),
        vtx_cxx = Var("userFloat('vtx_cxx')", float, doc = "error x of fitted vertex", precision=12),
        vtx_cyy = Var("userFloat('vtx_cyy')", float, doc = "error y of fitted vertex", precision=12),
        vtx_czz = Var("userFloat('vtx_czz')", float, doc = "error z of fitted vertex", precision=12),
        vtx_cyx = Var("userFloat('vtx_cyx')", float, doc = "error yx of fitted vertex", precision=12),
        vtx_czx = Var("userFloat('vtx_czx')", float, doc = "error zx of fitted vertex", precision=12),
        vtx_czy = Var("userFloat('vtx_czy')", float, doc = "error zy of fitted vertex", precision=12),
        # post-fit tracks/leptons
        #l1
        fit_l1_pt  = Var("userFloat('fitted_l1_pt')", float, doc = "post-fit leading mu pT"),
        fit_l1_eta = Var("userFloat('fitted_l1_eta')", float, doc = "post-fit leading mu eta"),
        fit_l1_phi = Var("userFloat('fitted_l1_phi')", float, doc = "post-fit leading mu phi"),
        #l2
        fit_l2_pt  = Var("userFloat('fitted_l2_pt')", float, doc = "post-fit subleading mu pT"),
        fit_l2_eta = Var("userFloat('fitted_l2_eta')", float, doc = "post-fit subleading mu eta"),
        fit_l2_phi = Var("userFloat('fitted_l2_phi')", float, doc = "post-fit subleading mu phi"),
        #trk1
        fit_trk1_pt  = Var("userFloat('fitted_trk1_pt')", float, doc = "post-fit leading track pT"),
        fit_trk1_eta = Var("userFloat('fitted_trk1_eta')", float, doc = "post-fit leading track eta"),
        fit_trk1_phi = Var("userFloat('fitted_trk1_phi')", float, doc = "post-fit leading track phi"),
        #trk2
        fit_trk2_pt  = Var("userFloat('fitted_trk2_pt')", float, doc = "post-fit subleading track pT"),
        fit_trk2_eta = Var("userFloat('fitted_trk2_eta')", float, doc = "post-fit subleading track eta"),
        fit_trk2_phi = Var("userFloat('fitted_trk2_phi')", float, doc = "post-fit subleading track phi"),
        # isolation 
        l1_iso04   = Var("userFloat('l1_iso04')", float, doc = "leading mu isolation DR<0.4", precision=12),
        l2_iso04   = Var("userFloat('l2_iso04')", float, doc = "subleading mu isolation DR<0.4", precision=12),
        trk1_iso04 = Var("userFloat('trk1_iso04')", float, doc = "leading track isolation DR<0.4", precision=12),
        trk2_iso04 = Var("userFloat('trk2_iso04')", float, doc = "subleading track isolation DR<0.4", precision=12),
        trk1_svip2d     = Var("userFloat('trk1_svip2d')", float, doc = "2D IP of the leading track wrt the dimuon vertex", precision=12),
        trk1_svip2d_err = Var("userFloat('trk1_svip2d_err')", float, doc = "uncertainty of 2D IP of the leading track wrt the dimuon vertex", precision=12),
        trk2_svip2d     = Var("userFloat('trk2_svip2d')", float, doc = "2D IP of the subleading track wrt the dimuon vertex", precision=12),
        trk2_svip2d_err = Var("userFloat('trk2_svip2d_err')", float, doc = "uncertainty of 2D IP of the subleading track wrt the dimuon vertex", precision=12),
        constraint_sv_prob     = Var("userFloat('constraint_sv_prob')", float, doc = "B vertex probability after the dimuon mass constraint", precision=12),
        constraint_pt   = Var("userFloat('constraint_pt')", float, doc = "B pt after the dimuon mass constraint", precision=12),
        constraint_eta  = Var("userFloat('constraint_eta')", float, doc = "B eta after the dimuon mass constraint", precision=12),
        constraint_phi  = Var("userFloat('constraint_phi')", float, doc = "B phi after the dimuon mass constraint", precision=12),
        constraint_mass_KK = Var("userFloat('constraint_mass_KK')", float, doc = "B mass with KK mass hypothesis after the dimuon mass constraint"),
        constraint_massErr_KK  = Var("userFloat('constraint_massErr_KK')", float, doc = "mass uncertainty for the KK mass hypothesis of the dimuon mass constraint", precision=12),
        constraint_mass_Kpi = Var("userFloat('constraint_mass_Kpi')", float, doc = "B mass with Kpi mass hypothesis after the dimuon mass constraint"),
        constraint_massErr_Kpi  = Var("userFloat('constraint_massErr_Kpi')", float, doc = "mass uncertainty for the Kpi mass hypothesis of the dimuon mass constraint", precision=12),
        constraint_mass_piK = Var("userFloat('constraint_mass_piK')", float, doc = "B mass with piK mass hypothesis after the dimuon mass constraint"),
        constraint_massErr_piK  = Var("userFloat('constraint_massErr_piK')", float, doc = "mass uncertainty for the piK mass hypothesis of the dimuon mass constraint", precision=12),
        constraint_mll  = Var("userFloat('constraint_mll')", float, doc = "dimuon mass after the dimuon mass constraint"),
    )
)

DiTrackTable = cms.EDProducer("SimpleCompositeCandidateFlatTableProducer",
    src  = cms.InputTag("BToTrkTrkMuMu:SelectedTrkTrk"),
    cut  = cms.string(""), #we should not filter on cross linked collections
    name = cms.string("DiTrack"),
    doc  = cms.string("slimmedDiTrack for BPark after basic selection"),
    singleton = cms.bool(False), # the number of entries is variable
    extension = cms.bool(False), # this is the main table for the muons
    variables = cms.PSet(
        CandVars,
        fit_mass_KK = Var("userFloat('fitted_mass_KK')", float, doc="mass of the ditrack candidate with leading->kaon, subleading->kaon mass hypothesis"),
        fit_mass_Kpi = Var("userFloat('fitted_mass_Kpi')", float, doc="mass of the ditrack candidate with leading->kaon, subleading->pion mass hypothesis"),
        fit_mass_piK = Var("userFloat('fitted_mass_piK')", float, doc="mass of the ditrack candidate with leading->pion, subleading->kaon mass hypothesis"),
        fit_pt   = Var("userFloat('fitted_pt')", float, doc="post-fit pt of the ditrack candidate"),
        fit_eta  = Var("userFloat('fitted_eta')", float, doc="post-fit eta of the ditrack candidate", precision=12),
        fit_phi  = Var("userFloat('fitted_phi')", float, doc="post-fit phi of the ditrack candidate", precision=12),
        svprob      = Var("userFloat('sv_prob')", float, doc="post-fit vertex probability of the ditrack candidate"),
        trk1_idx    = Var("userInt('trk1_idx')", int, doc="index of leading track to the Track collection"),
        trk2_idx    = Var("userInt('trk2_idx')", int, doc="index of subleading track to the Track collection"),
        vtx_x       = Var("userFloat('vtx_x')", float, doc="post-fit vertex x position", precision=12),
        vtx_y       = Var("userFloat('vtx_y')", float, doc="post-fit vertex y position", precision=12),
        vtx_z       = Var("userFloat('vtx_z')", float, doc="post-fit vertex x position", precision=12),
        l_xy        = Var("userFloat('l_xy')", float, doc="post-fit flight length"),
        l_xy_unc        = Var("userFloat('l_xy_unc')", float, doc="post-fit flight length uncertainty"),
        cos_theta_2D    = Var("userFloat('fitted_cos_theta_2D')", float, doc="post-fit cos"),
        sv_prob         = Var("userFloat('sv_prob')", float, doc="vertex probability"),
        sv_ndof         = Var("userFloat('sv_ndof')", float, doc="vertex ndof", precision=12),
        sv_chi2         = Var("userFloat('sv_chi2')", float, doc="vertex chi^2", precision=12),
        vtx_cxx = Var("userFloat('vtx_cxx')", float, doc="post-fit vertex xx uncertainty", precision=12),
        vtx_cyy = Var("userFloat('vtx_cyy')", float, doc="post-fit vertex yy uncertainty", precision=12),
        vtx_czz = Var("userFloat('vtx_czz')", float, doc="post-fit vertex zz uncertainty", precision=12),
        vtx_cyx = Var("userFloat('vtx_cyx')", float, doc="post-fit vertex yx uncertainty", precision=12),
        vtx_czx = Var("userFloat('vtx_czx')", float, doc="post-fit vertex zx uncertainty", precision=12),
        vtx_czy = Var("userFloat('vtx_czy')", float, doc="post-fit vertex zy uncertainty", precision=12)

    )
)


CountBToTrkTrkMuMu = cms.EDFilter("PATCandViewCountFilter",
    minNumber = cms.uint32(1),
    maxNumber = cms.uint32(999999),
    src       = cms.InputTag("BToTrkTrkMuMu")
)

########################### Sequencies  ############################
BToTrkTrkMuMuSequence = cms.Sequence( BToTrkTrkMuMu  )
BToTrkTrkMuMuTables   = cms.Sequence( BToTrkTrkMuMuTable + DiTrackTable )
