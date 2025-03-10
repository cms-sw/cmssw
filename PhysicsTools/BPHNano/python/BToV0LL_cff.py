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
    dileptonMassContraint = cms.bool(True)
)

LambdabToLambdaMuMu = cms.EDProducer(
    'BToV0LLBuilder',
    dileptons = cms.InputTag("MuMu:SelectedDiLeptons"),
    leptonTransientTracks = cms.InputTag('muonBPH', 'SelectedTransientMuons'),
    v0s = cms.InputTag('LambdaToProtonPi','SelectedV0Collection'),
    v0TransientTracks = cms.InputTag('LambdaToProtonPi', 'SelectedV0TransientCollection'),
    tracks = cms.InputTag("packedPFCandidates"),
    PUtracks = cms.InputTag('tracksBPH', 'SelectedTracks'),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    preVtxSelection  = cms.string('pt > 5.0 '
                                  '&& 4.5 < mass && mass < 6.5 '
                                  '&& userFloat("min_dr") > 0.03'),
    postVtxSelection = cms.string('4.9 < userFloat("fitted_mass") && userFloat("fitted_mass") < 6.3 '
                                  '&& userFloat("sv_prob") > 0.001 && -0.045<userFloat("v0_svip2d") && userFloat("v0_svip2d")<0.045'
                                  '&& userFloat("fitted_cos_theta_2D") > 0.9'),
    dileptonMassContraint = cms.bool(True)
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
        l1_idx      = Var("userInt('l1_idx')", int, doc = "leading muon index to the BPH muon collection"),
        l2_idx      = Var("userInt('l2_idx')", int, doc = "subleading muon index to the BPH muon collection"),
        V0_idx  = Var("userInt('v0_idx')", int, doc = "V0 index to the V0 collection"),
        min_dr      = Var("userFloat('min_dr')", float, doc = "minimum DeltaR between the V0 and the two muons"),
        max_dr      = Var("userFloat('max_dr')", float, doc = "maximum DeltaR between the V0 and the two muons"),
        # fit and vtx info
        chi2      = Var("userFloat('sv_chi2')", float, doc = "vertex chi^2 of the B candidate"),
        svprob    = Var("userFloat('sv_prob')", float, doc = "vertex probability of the B candidate"),
        l_xy      = Var("userFloat('l_xy')", float, doc = "post-fit vertex displacement on transverse plane wrt beamspot"),
        l_xy_unc  = Var("userFloat('l_xy_unc')", float, doc = "post-fit uncertainty of the vertex displacement on transverse plane wrt beamspot"),
        cos2D     = Var("userFloat('cos_theta_2D')", float, doc = "cos 2D of pre-fit candidate wrt beamspot"),
        fit_cos2D = Var("userFloat('fitted_cos_theta_2D')", float, doc = "cos 2D of fitted vertex wrt beamspot"),
        vtx_x     = Var("userFloat('vtx_x')", float, doc = "position x of fitted vertex"),
        vtx_y     = Var("userFloat('vtx_y')", float, doc = "position y of fitted vertex"),
        vtx_z     = Var("userFloat('vtx_z')", float, doc = "position z of fitted vertex"),
        vtx_cxx   = Var("userFloat('vtx_cxx')", float, doc = "error x of fitted vertex"),
        vtx_cyy   = Var("userFloat('vtx_cyy')", float, doc = "error y of fitted vertex"),
        vtx_czz   = Var("userFloat('vtx_czz')", float, doc = "error z of fitted vertex"),
        vtx_cyx   = Var("userFloat('vtx_cyx')", float, doc = "error yx of fitted vertex"),
        vtx_czx   = Var("userFloat('vtx_czx')", float, doc = "error zx of fitted vertex"),
        vtx_czy   = Var("userFloat('vtx_czy')", float, doc = "error zy of fitted vertex"),
        # post fit properties
        mll_fullfit     = Var("userFloat('fitted_mll')", float, doc = "post-fit mass of the two muons"),
        mV0_fullfit = Var("userFloat('fitted_v0_mass')", float, doc = "mass of the V0 candidate"),
        fit_mass        = Var("userFloat('fitted_mass')", float, doc = "post-fit mass of the B candidate"),
        fit_massErr     = Var("userFloat('fitted_massErr')", float, doc = "post-fit uncertainty of the mass of the B candidate"),
        fit_pt          = Var("userFloat('fitted_pt')", float, doc = "post-fit B pT"),
        fit_eta         = Var("userFloat('fitted_eta')", float, doc = "post-fit B eta"),
        fit_phi         = Var("userFloat('fitted_phi')", float, doc = "post-fit B phi"),
        # post-fit tracks/leptons
        #l1
        fit_l1_pt  = Var("userFloat('fitted_l1_pt')", float, doc = "post-fit leading mu pT"),
        fit_l1_eta = Var("userFloat('fitted_l1_eta')", float, doc = "post-fit leading mu eta"),
        fit_l1_phi = Var("userFloat('fitted_l1_phi')", float, doc = "post-fit leading mu phi"),
        #l2
        fit_l2_pt  = Var("userFloat('fitted_l2_pt')", float, doc = "post-fit subleading mu pT"),
        fit_l2_eta = Var("userFloat('fitted_l2_eta')", float, doc = "post-fit subleading mu eta"),
        fit_l2_phi = Var("userFloat('fitted_l2_phi')", float, doc = "post-fit subleading mu phi"),
        #V0
        fit_V0_pt  = Var("userFloat('fitted_v0_pt')", float, doc = "post-fit V0 pT"),
        fit_V0_eta = Var("userFloat('fitted_v0_eta')", float, doc = "post-fit V0 pT"),
        fit_V0_phi = Var("userFloat('fitted_v0_phi')", float, doc = "post-fit V0 pT"),
        # isolation and ipd2d
        V0_svip2d     = Var("userFloat('v0_svip2d')", float, doc = "2D IP of the V0 wrt the dimuon vertex"),
        V0_svip2d_err = Var("userFloat('v0_svip2d_err')", float, doc = "uncertainty of 2D IP of the V0 wrt the dimuon vertex"),
        l1_iso04          = Var("userFloat('l1_iso04')", float, doc = "leading muon isolation DR<0.4"),
        l2_iso04          = Var("userFloat('l2_iso04')", float, doc = "suleading muon isolation DR<0.4"),
        V0_iso04      = Var("userFloat('v0_iso04')", float, doc = "V0 isolation DR<0.4"),
        constraint_sv_prob     = Var("userFloat('constraint_sv_prob')", float, doc = "B vertex probability after the dimuon mass constraint"),
        constraint_pt   = Var("userFloat('constraint_pt')", float, doc = "B pt after the dimuon mass constraint"),
        constraint_eta  = Var("userFloat('constraint_eta')", float, doc = "B eta after the dimuon mass constraint"),
        constraint_phi  = Var("userFloat('constraint_phi')", float, doc = "B phi after the dimuon mass constraint"),
        constraint_mass = Var("userFloat('constraint_mass')", float, doc = "B mass after the dimuon mass constraint"),
        constraint_massErr  = Var("userFloat('constraint_massErr')", float, doc = "mass uncertainty of the dimuon mass constraint"),
        constraint_mll  = Var("userFloat('constraint_mll')", float, doc = "dimuon mass after the dimuon mass constraint"),
    )
)

LambdabToLambdaMuMuTable = BToKshortMuMuTable.clone(
    src = cms.InputTag("LambdabToLambdaMuMu"),
    name = cms.string("LambdabToLambdaMuMu"),
    doc = cms.string("LambdabToLambdaMuMu Variable")
)

CountBToKshortMuMu = cms.EDFilter("PATCandViewCountFilter",
    minNumber = cms.uint32(1),
    maxNumber = cms.uint32(999999),
    src       = cms.InputTag("BToKshortMuMu")
)

CountLambdabToLambdaMuMu = cms.EDFilter("PATCandViewCountFilter",
    minNumber = cms.uint32(1),
    maxNumber = cms.uint32(999999),
    src       = cms.InputTag("LambdabToLambdaMuMu")
)



########################### Sequencies  ############################
BToKshortMuMuSequence = cms.Sequence( BToKshortMuMu )
BToKshortMuMuTables   = cms.Sequence( BToKshortMuMuTable )

LambdabToLambdaMuMuSequence = cms.Sequence( LambdabToLambdaMuMu )
LambdabToLambdaMuMuTables  = cms.Sequence(LambdabToLambdaMuMuTable )



