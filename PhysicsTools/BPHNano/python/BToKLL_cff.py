import FWCore.ParameterSet.Config as cms
from PhysicsTools.BPHNano.common_cff import *

########################### B-> K ll ###########################

BToKmumu = cms.EDProducer(
    'BToTrkLLBuilder',
    dileptons = cms.InputTag("MuMu:SelectedDiLeptons"),
    leptonTransientTracks = cms.InputTag('muonBPH', 'SelectedTransientMuons'),
    dileptonKinVtxs = cms.InputTag('MuMu:SelectedDiLeptonKinVtxs'),
    kaons = cms.InputTag('tracksBPH', 'SelectedTracks'),
    kaonsTransientTracks = cms.InputTag('tracksBPH', 'SelectedTransientTracks'),
    trackMass = cms.double(493.677),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    PUtracks = cms.InputTag('tracksBPH', 'SelectedTracks'),
    preVtxSelection  = cms.string('pt > 5.0'
                                  '&& 4.5 < mass && mass < 6.'
                                  '&& userFloat("min_dr") > 0.03'),
    postVtxSelection = cms.string('4.7 < userFloat("fitted_mass") && userFloat("fitted_mass") < 6.' 
                                  '&& userFloat("sv_prob") > 0.001 && -0.045<userFloat("k_svip2d") && userFloat("k_svip2d")<0.045'
                                  '&& userFloat("fitted_cos_theta_2D") > 0.9'),
    dileptonMassContraint = cms.bool(True)
)

########################### Tables ###########################

BToKmumuTable = cms.EDProducer(
    'SimpleCompositeCandidateFlatTableProducer',
    src       = cms.InputTag("BToKmumu"),
    cut       = cms.string(""),
    name      = cms.string("BToKMuMu"),
    doc       = cms.string("BToKMuMu Variable"),
    singleton = cms.bool(False),
    extension = cms.bool(False),
    variables = cms.PSet(
        # pre-fit quantities
        CandVars,
        l1_idx = Var("userInt('l1_idx')", int, doc = "leading muon index to the BPH muon collection"),
        l2_idx = Var("userInt('l2_idx')", int, doc = "subleading muon index to the BPH muon collection"),
        k_idx  = Var("userInt('trk_idx')", int, doc = "track index to the BPH track collection"),
        minDR  = Var("userFloat('min_dr')", float, doc = "minimum DeltaR between the kaon and the two muons", precision=10),
        maxDR  = Var("userFloat('max_dr')", float, doc = "maximum DeltaR between the kaon and the two muons", precision=10),
        # fit and vtx info
        svprob   = Var("userFloat('sv_prob')", float, doc = "vertex probability of the B candidate", precision=10),
        l_xy     = Var("userFloat('l_xy')", float, doc = "post-fit vertex displacement on transverse plane wrt beamspot", precision=10),
        l_xy_unc = Var("userFloat('l_xy_unc')", float, doc = "post-fit vertex uncertainty of displacement on transverse plane wrt beamspot", precision=10),

        vtx_x   = Var("userFloat('vtx_x')", float, doc = "position x of fitted vertex", precision=10),
        vtx_y   = Var("userFloat('vtx_y')", float, doc = "position y of fitted vertex", precision=10),
        vtx_z   = Var("userFloat('vtx_z')", float, doc = "position z of fitted vertex", precision=10),
        vtx_cxx = Var("userFloat('vtx_cxx')", float, doc = "error x of fitted vertex", precision=10),
        vtx_cyy = Var("userFloat('vtx_cyy')", float, doc = "error y of fitted vertex", precision=10),
        vtx_czz = Var("userFloat('vtx_czz')", float, doc = "error z of fitted vertex", precision=10),
        vtx_cyx = Var("userFloat('vtx_cyx')", float, doc = "error yx of fitted vertex", precision=10),
        vtx_czx = Var("userFloat('vtx_czx')", float, doc = "error zx of fitted vertex", precision=10),
        vtx_czy = Var("userFloat('vtx_czy')", float, doc = "error zy of fitted vertex", precision=10),
        # Mll
        mll_fullfit = Var("userFloat('fitted_mll')", float, doc = "post-fit mass of the two muons", precision=10),
        # Cos(theta)
        cos2D     = Var("userFloat('cos_theta_2D')", float, doc = "cos 2D of pre-fit candidate wrt beamspot", precision=10),
        fit_cos2D = Var("userFloat('fitted_cos_theta_2D')", float, doc = "cos 2D of fitted vertex wrt beamspot", precision=10),
        # post-fit momentum
        fit_mass    = Var("userFloat('fitted_mass')", float, doc = "post-fit mass of the B candidate", precision=10),
        fit_massErr = Var("userFloat('fitted_massErr')", float, doc = "post-fit uncertainty of the mass of the B candidate", precision=10),
        fit_pt      = Var("userFloat('fitted_pt')", float, doc = "post-fit B pT", precision=10),
        fit_eta     = Var("userFloat('fitted_eta')", float, doc = "post-fit B eta", precision=10),
        fit_phi     = Var("userFloat('fitted_phi')", float, doc = "post-fit B phi", precision=10),
        fit_l1_pt   = Var("userFloat('fitted_l1_pt')", float, doc = "post-fit leading mu pT", precision=10),
        fit_l1_eta  = Var("userFloat('fitted_l1_eta')", float, doc = "post-fit leading mu eta", precision=10),
        fit_l1_phi  = Var("userFloat('fitted_l1_phi')", float, doc = "post-fit leading mu phi", precision=10),
        fit_l2_pt   = Var("userFloat('fitted_l2_pt')", float, doc = "post-fit subleading mu pT", precision=10),
        fit_l2_eta  = Var("userFloat('fitted_l2_eta')", float, doc = "post-fit subleading mu eta", precision=10),
        fit_l2_phi  = Var("userFloat('fitted_l2_phi')", float, doc = "post-fit subleading mu phi", precision=10),
        fit_k_pt    = Var("userFloat('fitted_trk_pt')", float, doc = "post-fit track pT", precision=10),
        fit_k_eta   = Var("userFloat('fitted_trk_eta')", float, doc = "post-fit track eta", precision=10),
        fit_k_phi   = Var("userFloat('fitted_trk_phi')", float, doc = "post-fit track phi", precision=10),
        k_svip2d    = Var("userFloat('k_svip2d')", float, doc = "2D IP of the track wrt the dimuon vertex", precision=10),
        k_svip2d_err = Var("userFloat('k_svip2d_err')", float, doc = "uncertainty of 2D IP of the track wrt the dimuon vertex", precision=10),
        l1_iso04    = Var("userFloat('l1_iso04')", float, doc = "leading mu isolation DR<0.4", precision=10),
        l2_iso04    = Var("userFloat('l2_iso04')", float, doc = "subleading mu isolation DR<0.4", precision=10),
        k_iso04     = Var("userFloat('trk_iso04')", float, doc = "track isolation DR<0.4", precision=10),
        constraint_sv_prob     = Var("userFloat('constraint_sv_prob')", float, doc = "B vertex probability after the dimuon mass constraint", precision=10),
        constraint_pt   = Var("userFloat('constraint_pt')", float, doc = "B pt after the dimuon mass constraint", precision=10),
        constraint_eta  = Var("userFloat('constraint_eta')", float, doc = "B eta after the dimuon mass constraint", precision=10),
        constraint_phi  = Var("userFloat('constraint_phi')", float, doc = "B phi after the dimuon mass constraint", precision=10),
        constraint_mass = Var("userFloat('constraint_mass')", float, doc = "B mass after the dimuon mass constraint", precision=10),
        constraint_massErr  = Var("userFloat('constraint_massErr')", float, doc = "mass uncertainty of the dimuon mass constraint", precision=10),
        constraint_mll  = Var("userFloat('constraint_mll')", float, doc = "dimuon mass after the dimuon mass constraint", precision=10),
    )
)


CountBToKmumu = cms.EDFilter("PATCandViewCountFilter",
    minNumber = cms.uint32(1),
    maxNumber = cms.uint32(999999),
    src       = cms.InputTag("BToKmumu")
)    

########################### Sequencies  ############################
BToKMuMuSequence = cms.Sequence(BToKmumu)
BToKMuMuTables   = cms.Sequence(BToKmumuTable)
