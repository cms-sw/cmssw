import FWCore.ParameterSet.Config as cms
from PhysicsTools.BPHNano.common_cff import *


########################### B-> K* ll ##########################

BToChargedKstarsMuMu = cms.EDProducer(
    'BToV0TrkLLBuilder',
    dileptons = cms.InputTag('MuMu:SelectedDiLeptons'),
    leptonTransientTracks = cms.InputTag('muonBPH', 'SelectedTransientMuons'),
    V0s_ttracks = cms.InputTag('KshortToPiPi','SelectedV0TransientCollection'),
    V0s = cms.InputTag('KshortToPiPi','SelectedV0Collection'),
    pions = cms.InputTag('tracksBPH', 'SelectedTracks'),
    pionsTransientTracks = cms.InputTag('tracksBPH', 'SelectedTransientTracks'),
    tracks = cms.InputTag("packedPFCandidates"),
    lostTracks = cms.InputTag("lostTracks"),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    offlinePrimaryVertexSrc = cms.InputTag('offlineSlimmedPrimaryVertices'),
    preVtxSelection = cms.string('userFloat("pi_min_dr")>0.03 && abs(charge)==1 && 4.<mass && mass<7.0 && 0.7<userFloat("mIntermediate_unfitted") && userFloat("mIntermediate_unfitted")<1.1 && userFloat("V0pi_dr")<1.6 && userFloat("ll_pi_deltaR")<1.8 && userFloat("ll_V0_deltaR")<1.8'),
    postVtxSelection = cms.string('userFloat("sv_prob") > 1.e-3 && userFloat("fitted_cos_theta_2D") >= 0.90 &&'
    ' (4.5<userFloat("fitted_mass") && userFloat("fitted_mass")<6.0 ) && 0.7<userFloat("fitted_KstarPlus_mass") && userFloat("fitted_KstarPlus_mass")<1.1 && -0.045<userFloat("pi_svip2d") && userFloat("pi_svip2d")<0.045 && -0.045<userFloat("V0_svip2d") && userFloat("V0_svip2d")<0.045'
    ),
)

########################### Xib-> Xi- ll ##########################

XibToXiMuMu = cms.EDProducer(
    'BToV0TrkDisplacedLLBuilder',
    dileptons = cms.InputTag('MuMu:SelectedDiLeptons'),
    leptonTransientTracks = cms.InputTag('muonBPH', 'SelectedTransientMuons'),
    V0s_ttracks = cms.InputTag('LambdaToProtonPi','SelectedV0TransientCollection'),
    V0s = cms.InputTag('LambdaToProtonPi','SelectedV0Collection'),
    pions = cms.InputTag('tracksBPH', 'SelectedTracks'),
    pionsTransientTracks = cms.InputTag('tracksBPH', 'SelectedTransientTracks'),
    tracks = cms.InputTag("packedPFCandidates"),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    offlinePrimaryVertexSrc = cms.InputTag('offlineSlimmedPrimaryVertices'),
    preVtxSelection = cms.string('userFloat("pi_min_dr")>0.03 && abs(charge)==1 && 4.<mass && mass<7.0 && 1.240<userFloat("mIntermediate_unfitted") && userFloat("mIntermediate_unfitted")<1.450 && userFloat("V0pi_dr")<1.6 && userFloat("ll_pi_deltaR")<1.8 && userFloat("ll_V0_deltaR")<1.8'),
    postVtxSelection = cms.string('userFloat("sv_prob") > 1.e-3 && userFloat("fitted_cos_theta_2D") >= 0.90 &&'
    ' (5.0<userFloat("fitted_mass") && userFloat("fitted_mass")<6.5 ) && 1.240<userFloat("Xi_mass") && userFloat("Xi_mass")<1.450'
    ),
)



BToChargedKstarsMuMuTable = cms.EDProducer(
    'SimpleCompositeCandidateFlatTableProducer',
    src = cms.InputTag("BToChargedKstarsMuMu"),
    cut = cms.string(""),
    name = cms.string("BToChargedKstarMuMu"),
    doc = cms.string("BToChargedKstarMuMu Variable"),
    singleton=cms.bool(False),
    extension=cms.bool(False),
    variables=cms.PSet(
      # pre-fit quantities
      CandVars,
      pi_idx  = Var("userInt('pi_idx')", int, doc = "track index to the BPH track collection"),
      V0_idx  = Var("userInt('V0_idx')", int, doc = "V0 index to the BPH V0 collection"),
      l1_idx = Var("userInt('l1_idx')", int, doc = "leading muon index to the BPH muon collection"),
      l2_idx = Var("userInt('l2_idx')", int, doc = "subleading muon index to the BPH muon collection"),
      ll_idx  = Var("userInt('ll_idx')", int, doc = "dilepton index to the MuMu collection"),
      V0trk_dz = Var("userFloat('V0trk_dz')", float, doc = "Dz at DCA between the V0 and the dilepton pair", precision=12),
      V0_dz  = Var("userFloat('V0_dz')", float, doc = "Dz at DCA between the V0 and the dilepton pair", precision=12),
      V0pi_dr = Var("userFloat('V0pi_dr')", float, doc = "DeltaR between the V0 and the pion", precision=12),
      V0_min_dr = Var("userFloat('V0_min_dr')", float, doc = "minimum DeltaR between the V0 and the two muons", precision=12),
      V0_max_dr = Var("userFloat('V0_max_dr')", float, doc = "maximum DeltaR between the V0 and the two muons", precision=12),
      pi_min_dr = Var("userFloat('pi_min_dr')", float, doc = "minimum DeltaR between the pion and the two muons", precision=12),
      pi_max_dr = Var("userFloat('pi_max_dr')", float, doc = "maximum DeltaR between the pion and the two muons", precision=12),
      #fit_charge = uint('fitted_charge'),
      #unfit_B_mass = ufloat('unfitted_B_mass'),
      sv_chi2 = Var("userFloat('sv_chi2')", float, doc = "chi^2 of the Xi_b fitted candidate", precision=12),
      sv_ndof = Var("userFloat('sv_ndof')", float, doc = "ndof of the Xi_b fitted candidate", precision=12),
      sv_prob = Var("userFloat('sv_prob')", float, doc = "vertex probability of the Xi_b candidate"),
      fit_KstarPlus_mass = Var("userFloat('fitted_KstarPlus_mass')", float, doc = "post-fit K*+ mass"),
      fit_KstarPlus_pt = Var("userFloat('fitted_KstarPlus_pt')", float, doc = "post-fit K*+ pT"),
      fit_KstarPlus_eta = Var("userFloat('fitted_KstarPlus_eta')", float, doc = "post-fit K*+ eTa"),
      fit_KstarPlus_phi = Var("userFloat('fitted_KstarPlus_phi')", float, doc = "post-fit K*+ phi"),
      fit_mass = Var("userFloat('fitted_mass')", float, doc = "post-fit mass of the Xi_b candidate"),
      fit_massErr = Var("userFloat('fitted_massErr')", float, doc = "post-fit uncertainty of the mass of the Xi_b candidate", precision=12),
      mll_fullfit = Var("userFloat('fitted_mll_mass')", float, doc = "post-fit mass of the two muons"),
      fit_pt = Var("userFloat('fitted_pt')", float, doc = "post-fit Xi_b pT"),
      fit_eta = Var("userFloat('fitted_eta')", float, doc = "post-fit Xi_b eta"),
      fit_phi = Var("userFloat('fitted_phi')", float, doc = "post-fit Xi_b phi"),
      dca = Var("userFloat('dca')", float, doc = "DCA of B candidate wrt to beamspot", precision=12),
      dcaErr = Var("userFloat('dcaErr')", float, doc = "Error DCA of B candidate wrt to beamspot", precision=12),
      vtx_x = Var("userFloat('vtx_x')", float, doc = "position x of Xi_b fitted vertex", precision=12),
      vtx_y = Var("userFloat('vtx_y')", float, doc = "position y of Xi_b fitted vertex", precision=12),
      vtx_z = Var("userFloat('vtx_z')", float, doc = "position z of Xi_b fitted vertex", precision=12),
      vtx_ex = Var("userFloat('vtx_ex')", float, doc = "uncertainty for position x of Xi_b fitted vertex", precision=12),
      vtx_ey = Var("userFloat('vtx_ey')", float, doc = "uncertainty for position y of Xi_b fitted vertex", precision=12),
      vtx_ez = Var("userFloat('vtx_ez')", float, doc = "uncertainty for position z of Xi_b fitted vertex", precision=12),
      fit_pi_pt = Var("userFloat('fitted_pi_pt')", float, doc = "post-fit pion pt"),
      fit_pi_eta = Var("userFloat('fitted_pi_eta')", float, doc = "post-fit pion eta"),
      fit_pi_phi = Var("userFloat('fitted_pi_phi')", float, doc = "post-fit pion phi"),
      fit_V0_pt = Var("userFloat('fitted_V0_pt')", float, doc = "post-fit V0 pt"),
      fit_V0_eta = Var("userFloat('fitted_V0_eta')", float, doc = "post-fit V0 eta"),
      fit_V0_phi = Var("userFloat('fitted_V0_phi')", float, doc = "post-fit V0 phi"),
      fit_l1_pt = Var("userFloat('fitted_l1_pt')", float, doc = "post-fit leading mu pT"),
      fit_l1_eta = Var("userFloat('fitted_l1_eta')", float, doc = "post-fit leading mu eta"),
      fit_l1_phi = Var("userFloat('fitted_l1_phi')", float, doc = "post-fit leading mu phi"),
      fit_l2_pt = Var("userFloat('fitted_l2_pt')", float, doc = "post-fit subleading mu pT"),
      fit_l2_eta = Var("userFloat('fitted_l2_eta')", float, doc = "post-fit subleading mu eta"),
      fit_l2_phi = Var("userFloat('fitted_l2_phi')", float, doc = "post-fit subleading mu phi"),
      fit_cos_theta_2D = Var("userFloat('fitted_cos_theta_2D')", float, doc = "cos 2D of fitted vertex wrt beamspot for the Xi candidate"),
      l_xy = Var("userFloat('l_xy')", float, doc = "post-fit vertex displacement on transverse plane wrt beamspot"),
      l_xy_unc = Var("userFloat('l_xy_unc')", float, doc = "post-fit vertex uncertainty of displacement on transverse plane wrt beamspot"),
      pi_svip2d = Var("userFloat('pi_svip2d')", float, doc = "2D IP of the track wrt the dimuon vertex", precision=12),
      pi_svip2d_err = Var("userFloat('pi_svip2d_err')", float, doc = "uncertainty of 2D IP of the track wrt the dimuon vertex", precision=12),
      pi_svip3d = Var("userFloat('pi_svip3d')", float, doc = "3D IP of the track wrt the dimuon vertex", precision=12),
      pi_svip3d_err = Var("userFloat('pi_svip3d_err')", float, doc = "uncertainty of 3D IP of the track wrt the dimuon vertex", precision=12),
      V0_svip2d = Var("userFloat('V0_svip2d')", float, doc = "2D IP of the V0 wrt the dimuon vertex", precision=12),
      V0_svip2d_err = Var("userFloat('V0_svip2d_err')", float, doc = "uncertainty of 2D IP of the V0 wrt the dimuon vertex", precision=12),
      V0_svip3d = Var("userFloat('V0_svip3d')", float, doc = "3D IP of the V0 wrt the dimuon vertex", precision=12),
      V0_svip3d_err = Var("userFloat('V0_svip3d_err')", float, doc = "uncertainty of 3D IP of the V0 wrt the dimuon vertex", precision=12),
      ll_pi_deltaR = Var("userFloat('ll_pi_deltaR')", float, doc = "Delta R between pion and dilepton pair", precision=12),
      ll_V0_deltaR = Var("userFloat('ll_V0_deltaR')", float, doc = "Delta R between V0 and dilepton pair", precision=12),
      constraint_chi2 = Var("userFloat('sv_chi2_withMC')", float, doc = "chi^2 of the Xi_b fitted candidate with mass constraint", precision=12),
      constraint_ndof = Var("userFloat('sv_ndof_withMC')", float, doc = "ndof of the Xi_b fitted candidate with mass constraint", precision=12),
      constraint_prob = Var("userFloat('sv_prob_withMC')", float, doc = "vertex probability of the Xi_b candidate with mass constraint", precision=12),
      constraint_l_xy = Var("userFloat('l_xy_withMC')", float, doc = "post-fit vertex displacement on transverse plane wrt beamspot with mass constraint", precision=12),
      constraint_l_xy_unc = Var("userFloat('l_xy_unc_withMC')", float, doc = "post-fit vertex uncertainty of displacement on transverse plane wrt beamspot with mass constraint", precision=12),
      constraint_vtx_x = Var("userFloat('vtx_x_withMC')", float, doc = "position x of Xi_b fitted vertex with mass constraint", precision=12),
      constraint_vtx_y = Var("userFloat('vtx_y_withMC')", float, doc = "position y of Xi_b fitted vertex with mass constraint", precision=12),
      constraint_vtx_z = Var("userFloat('vtx_z_withMC')", float, doc = "position z of Xi_b fitted vertex with mass constraint", precision=12),
      constraint_vtx_ex = Var("userFloat('vtx_ex_withMC')", float, doc = "uncertainty for position x of Xi_b fitted vertex with mass constraint", precision=12),
      constraint_vtx_ey = Var("userFloat('vtx_ey_withMC')", float, doc = "uncertainty for position y of Xi_b fitted vertex with mass constraint", precision=12),
      constraint_vtx_ez = Var("userFloat('vtx_ez_withMC')", float, doc = "uncertainty for position z of Xi_b fitted vertex with mass constraint", precision=12),
      constraint_fit_cos_theta_2D = Var("userFloat('fitted_cos_theta_2D_withMC')", float, doc = "cos 2D of fitted vertex wrt beamspot for the Xi candidate with mass constraint", precision=12),
      constraint_fit_mass = Var("userFloat('fitted_mass_withMC')", float, doc = "post-fit mass of the Xi_b candidate with mass constraint"),
      constraint_fit_massErr = Var("userFloat('fitted_massErr_withMC')", float, doc = "post-fit mass uncertainty of the Xi_b candidate with mass constraint", precision=12),
      constraint_fit_pt = Var("userFloat('fitted_pt_withMC')", float, doc = "post-fit Xi_b pT with mass constraint", precision=12),
      constraint_fit_eta = Var("userFloat('fitted_eta_withMC')", float, doc = "post-fit Xi_b eta with mass constraint", precision=12),
      constraint_fit_phi = Var("userFloat('fitted_phi_withMC')", float, doc = "post-fit Xi_b phi with mass constraint", precision=12),
      constraint_fit_l1_pt = Var("userFloat('fitted_l1_pt_withMC')", float, doc = "post-fit leading mu pT with mass constraint", precision=12),
      constraint_fit_l1_eta = Var("userFloat('fitted_l1_eta_withMC')", float, doc = "post-fit leading mu eta with mass constraint", precision=12),
      constraint_fit_l1_phi = Var("userFloat('fitted_l1_phi_withMC')", float, doc = "post-fit leading mu phi with mass constraint", precision=12),
      constraint_fit_l2_pt = Var("userFloat('fitted_l2_pt_withMC')", float, doc = "post-fit subleading mu pT with mass constraint", precision=12),
      constraint_fit_l2_eta = Var("userFloat('fitted_l2_eta_withMC')", float, doc = "post-fit subleading mu eta with mass constraint", precision=12),
      constraint_fit_l2_phi = Var("userFloat('fitted_l2_phi_withMC')", float, doc = "post-fit subleading mu phi with mass constraint", precision=12),
      constraint_fit_V0_pt = Var("userFloat('fitted_V0_pt_withMC')", float, doc = "post-fit V0 pT with mass constraint", precision=12),
      constraint_fit_V0_eta = Var("userFloat('fitted_V0_eta_withMC')", float, doc = "post-fit V0 eta with mass constraint", precision=12),
      constraint_fit_V0_phi = Var("userFloat('fitted_V0_phi_withMC')", float, doc = "post-fit V0 phi with mass constraint", precision=12),
      constraint_fit_pi_pt = Var("userFloat('fitted_pi_pt_withMC')", float, doc = "post-fit pion pT with mass constraint", precision=12),
      constraint_fit_pi_eta = Var("userFloat('fitted_pi_eta_withMC')", float, doc = "post-fit pion eta with mass constraint", precision=12),
      constraint_fit_pi_phi = Var("userFloat('fitted_pi_phi_withMC')", float, doc = "post-fit pion phi with mass constraint", precision=12),
    )
)



XibToXiMuMuTable = cms.EDProducer(
    'SimpleCompositeCandidateFlatTableProducer',
    src = cms.InputTag("XibToXiMuMu"),
    cut = cms.string(""),
    name = cms.string("XibToXiMuMu"),
    doc = cms.string("XibToXiMuMu Variable"),
    singleton=cms.bool(False),
    extension=cms.bool(False),
    variables=cms.PSet(
      # pre-fit quantities
      CandVars,
      pi_idx  = Var("userInt('pi_idx')", int, doc = "track index to the BPH track collection"),
      V0_idx  = Var("userInt('V0_idx')", int, doc = "V0 index to the BPH V0 collection"),    
      l1_idx = Var("userInt('l1_idx')", int, doc = "leading muon index to the BPH muon collection"),
      l2_idx = Var("userInt('l2_idx')", int, doc = "subleading muon index to the BPH muon collection"),
      ll_idx  = Var("userInt('ll_idx')", int, doc = "dilepton index to the MuMu collection"),
      V0pi_dr = Var("userFloat('V0pi_dr')", float, doc = "DeltaR between the V0 and the pion", precision=12),
      V0_min_dr = Var("userFloat('V0_min_dr')", float, doc = "minimum DeltaR between the V0 and the two muons", precision=12),
      V0_max_dr = Var("userFloat('V0_max_dr')", float, doc = "maximum DeltaR between the V0 and the two muons", precision=12),
      pi_min_dr = Var("userFloat('pi_min_dr')", float, doc = "minimum DeltaR between the pion and the two muons", precision=12),
      pi_max_dr = Var("userFloat('pi_max_dr')", float, doc = "maximum DeltaR between the pion and the two muons", precision=12),
      #fit_charge = uint('fitted_charge'),
      #unfit_B_mass = ufloat('unfitted_B_mass'),
      sv_chi2 = Var("userFloat('sv_chi2')", float, doc = "chi^2 of the Xi_b fitted candidate", precision=12),
      sv_ndof = Var("userFloat('sv_ndof')", float, doc = "ndof of the Xi_b fitted candidate", precision=12),
      sv_prob = Var("userFloat('sv_prob')", float, doc = "vertex probability of the Xi_b candidate", precision=12),
      Xi_sv_chi2 = Var("userFloat('Xi_sv_chi2')", float, doc = "chi^2 of the Xi fitted candidate", precision=12),
      Xi_sv_ndof = Var("userFloat('Xi_sv_ndof')", float, doc = "ndof of the Xi fitted candidate", precision=12),
      Xi_sv_prob = Var("userFloat('Xi_sv_prob')", float, doc = "vertex probability of the Xi candidate"),
      Xi_vtx_x = Var("userFloat('Xi_vtx_x')", float, doc = "position x of Xi fitted vertex", precision=12),
      Xi_vtx_y = Var("userFloat('Xi_vtx_y')", float, doc = "position y of Xi fitted vertex", precision=12),
      Xi_vtx_z = Var("userFloat('Xi_vtx_z')", float, doc = "position z of Xi fitted vertex", precision=12),
      Xi_vtx_ex = Var("userFloat('Xi_vtx_ex')", float, doc = "uncertainty for position x of Xi fitted vertex", precision=12),
      Xi_vtx_ey = Var("userFloat('Xi_vtx_ey')", float, doc = "uncertainty for position y of Xi fitted vertex", precision=12),
      Xi_vtx_ez = Var("userFloat('Xi_vtx_ez')", float, doc = "uncertainty for position z of Xi fitted vertex", precision=12),
      Xi_fit_cos_theta_2D = Var("userFloat('Xi_fitted_cos_theta_2D')", float, doc = "cos 2D of fitted vertex wrt beamspot for the Xi candidate"),
      Xi_l_xy = Var("userFloat('Xi_l_xy')", float, doc = "post-fit vertex displacement on transverse plane wrt beamspot for the Xi candidate"),
      Xi_l_xy_unc = Var("userFloat('Xi_l_xy_unc')", float, doc = "post-fit vertex uncertainty of displacement on transverse plane wrt beamspot for the Xi candidate"),
      Xi_mass = Var("userFloat('Xi_mass')", float, doc = "post-fit mass of the Xi candidate"),
      Xi_massErr = Var("userFloat('Xi_massErr')", float, doc = "uncertainty for post-fit mass of the Xi candidate", precision=12),
      fit_mass = Var("userFloat('fitted_mass')", float, doc = "post-fit mass of the B candidate"),
      fit_massErr = Var("userFloat('fitted_massErr')", float, doc = "post-fit uncertainty of the mass of the B candidate", precision=12),
      mll_fullfit = Var("userFloat('fitted_mll_mass')", float, doc = "post-fit mass of the two muons"),
      fit_pt = Var("userFloat('fitted_pt')", float, doc = "post-fit B pT"),
      fit_eta = Var("userFloat('fitted_eta')", float, doc = "post-fit B eta"),
      fit_phi = Var("userFloat('fitted_phi')", float, doc = "post-fit B phi"),
      dca = Var("userFloat('dca')", float, doc = "DCA of B candidate wrt to beamspot", precision=12),
      dcaErr = Var("userFloat('dcaErr')", float, doc = "Error DCA of B candidate wrt to beamspot", precision=12),
      vtx_x = Var("userFloat('vtx_x')", float, doc = "position x of Xi_b fitted vertex", precision=12),
      vtx_y = Var("userFloat('vtx_y')", float, doc = "position y of Xi_b fitted vertex", precision=12),
      vtx_z = Var("userFloat('vtx_z')", float, doc = "position z of Xi_b fitted vertex", precision=12),
      vtx_ex = Var("userFloat('vtx_ex')", float, doc = "uncertainty for position x of Xi_b fitted vertex", precision=12),
      vtx_ey = Var("userFloat('vtx_ey')", float, doc = "uncertainty for position y of Xi_b fitted vertex", precision=12),
      vtx_ez = Var("userFloat('vtx_ez')", float, doc = "uncertainty for position z of Xi_b fitted vertex", precision=12),
      fit_l1_pt = Var("userFloat('fitted_l1_pt')", float, doc = "post-fit leading mu pT"),
      fit_l1_eta = Var("userFloat('fitted_l1_eta')", float, doc = "post-fit leading mu eta"),
      fit_l1_phi = Var("userFloat('fitted_l1_phi')", float, doc = "post-fit leading mu phi"),
      fit_l2_pt = Var("userFloat('fitted_l2_pt')", float, doc = "post-fit subleading mu pT"),
      fit_l2_eta = Var("userFloat('fitted_l2_eta')", float, doc = "post-fit subleading mu eta"),
      fit_l2_phi = Var("userFloat('fitted_l2_phi')", float, doc = "post-fit subleading mu phi"),
      fit_Xi_pt = Var("userFloat('fitted_Xi_pt')", float, doc = "post-fit Xi pT"),
      fit_Xi_eta = Var("userFloat('fitted_Xi_eta')", float, doc = "post-fit Xi eta"),
      fit_Xi_phi = Var("userFloat('fitted_Xi_phi')", float, doc = "post-fit Xi phi"),
      fit_cos_theta_2D = Var("userFloat('fitted_cos_theta_2D')", float, doc = "cos 2D of fitted vertex wrt beamspot for the Xi candidate"),
      l_xy = Var("userFloat('l_xy')", float, doc = "post-fit vertex displacement on transverse plane wrt beamspot"),
      l_xy_unc = Var("userFloat('l_xy_unc')", float, doc = "post-fit vertex uncertainty of displacement on transverse plane wrt beamspot"),
      Xi_svip2d = Var("userFloat('Xi_svip2d')", float, doc = "2D IP of the track wrt the dimuon vertex", precision=12),
      Xi_svip2d_err = Var("userFloat('Xi_svip2d_err')", float, doc = "uncertainty of 2D IP of the track wrt the dimuon vertex", precision=12),
      Xi_svip3d = Var("userFloat('Xi_svip3d')", float, doc = "3D IP of the track wrt the dimuon vertex", precision=12),
      Xi_svip3d_err = Var("userFloat('Xi_svip3d_err')", float, doc = "uncertainty of 3D IP of the track wrt the dimuon vertex", precision=12),
      ll_pi_deltaR = Var("userFloat('ll_pi_deltaR')", float, doc = "Delta R between pion and dilepton pair", precision=12),
      ll_V0_deltaR = Var("userFloat('ll_V0_deltaR')", float, doc = "Delta R between V0 and dilepton pair", precision=12),
      constraint_chi2 = Var("userFloat('sv_chi2_withMC')", float, doc = "chi^2 of the Xi_b fitted candidate with mass constraint", precision=12),
      constraint_ndof = Var("userFloat('sv_ndof_withMC')", float, doc = "ndof of the Xi_b fitted candidate with mass constraint", precision=12),
      constraint_prob = Var("userFloat('sv_prob_withMC')", float, doc = "vertex probability of the Xi_b candidate with mass constraint", precision=12),
      constraint_l_xy = Var("userFloat('l_xy_withMC')", float, doc = "post-fit vertex displacement on transverse plane wrt beamspot with mass constraint", precision=12),
      constraint_l_xy_unc = Var("userFloat('l_xy_unc_withMC')", float, doc = "post-fit vertex uncertainty of displacement on transverse plane wrt beamspot with mass constraint", precision=12),
      constraint_vtx_x = Var("userFloat('vtx_x_withMC')", float, doc = "position x of Xi_b fitted vertex with mass constraint", precision=12),
      constraint_vtx_y = Var("userFloat('vtx_y_withMC')", float, doc = "position y of Xi_b fitted vertex with mass constraint", precision=12),
      constraint_vtx_z = Var("userFloat('vtx_z_withMC')", float, doc = "position z of Xi_b fitted vertex with mass constraint", precision=12),
      constraint_vtx_ex = Var("userFloat('vtx_ex_withMC')", float, doc = "uncertainty for position x of Xi_b fitted vertex with mass constraint", precision=12),
      constraint_vtx_ey = Var("userFloat('vtx_ey_withMC')", float, doc = "uncertainty for position y of Xi_b fitted vertex with mass constraint", precision=12),
      constraint_vtx_ez = Var("userFloat('vtx_ez_withMC')", float, doc = "uncertainty for position z of Xi_b fitted vertex with mass constraint", precision=12),
      constraint_fit_cos_theta_2D = Var("userFloat('fitted_cos_theta_2D_withMC')", float, doc = "cos 2D of fitted vertex wrt beamspot for the Xi candidate with mass constraint", precision=12),
      constraint_fit_mass = Var("userFloat('fitted_mass_withMC')", float, doc = "post-fit mass of the B candidate with mass constraint"),
      constraint_fit_massErr = Var("userFloat('fitted_massErr_withMC')", float, doc = "post-fit mass uncertainty of the B candidate with mass constraint", precision=12),
      constraint_fit_pt = Var("userFloat('fitted_pt_withMC')", float, doc = "post-fit B pT with mass constraint", precision=12),
      constraint_fit_eta = Var("userFloat('fitted_eta_withMC')", float, doc = "post-fit B eta with mass constraint", precision=12),
      constraint_fit_phi = Var("userFloat('fitted_phi_withMC')", float, doc = "post-fit B phi with mass constraint", precision=12),
      constraint_fit_l1_pt = Var("userFloat('fitted_l1_pt_withMC')", float, doc = "post-fit leading mu pT with mass constraint", precision=12),
      constraint_fit_l1_eta = Var("userFloat('fitted_l1_eta_withMC')", float, doc = "post-fit leading mu eta with mass constraint", precision=12),
      constraint_fit_l1_phi = Var("userFloat('fitted_l1_phi_withMC')", float, doc = "post-fit leading mu phi with mass constraint", precision=12),
      constraint_fit_l2_pt = Var("userFloat('fitted_l2_pt_withMC')", float, doc = "post-fit subleading mu pT with mass constraint", precision=12),
      constraint_fit_l2_eta = Var("userFloat('fitted_l2_eta_withMC')", float, doc = "post-fit subleading mu eta with mass constraint", precision=12),
      constraint_fit_l2_phi = Var("userFloat('fitted_l2_phi_withMC')", float, doc = "post-fit subleading mu phi with mass constraint", precision=12),
      constraint_fit_Xi_pt = Var("userFloat('fitted_Xi_pt_withMC')", float, doc = "post-fit Xi pT with mass constraint", precision=12),
      constraint_fit_Xi_eta = Var("userFloat('fitted_Xi_eta_withMC')", float, doc = "post-fit Xi eta with mass constraint", precision=12),
      constraint_fit_Xi_phi = Var("userFloat('fitted_Xi_phi_withMC')", float, doc = "post-fit Xi phi with mass constraint", precision=12),


    )
)

CountBToChargedKstarMuMu= cms.EDFilter("PATCandViewCountFilter",
    minNumber = cms.uint32(1),
    src = cms.InputTag("BToChargedKstarsMuMu"),
)

CountXibToXiMuMu= cms.EDFilter("PATCandViewCountFilter",
    minNumber = cms.uint32(0),
    src = cms.InputTag("XibToXiMuMu"),
)


BToChargedKstarMuMuSequence = cms.Sequence(
    BToChargedKstarsMuMu
)

XibToXiMuMuSequence = cms.Sequence(
    XibToXiMuMu
)










