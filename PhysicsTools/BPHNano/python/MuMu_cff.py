import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *

########################### Selections ###########################

MuMu = cms.EDProducer(
    'DiMuonBuilder',
    src = cms.InputTag('muonBPH', 'SelectedMuons'),
    transientTracksSrc = cms.InputTag('muonBPH', 'SelectedTransientMuons'),
    lep1Selection = cms.string('pt > 4.0 && abs(eta) < 2.4 && isLooseMuon && isGlobalMuon'),
    lep2Selection = cms.string('pt > 3.0 && abs(eta) < 2.4 && isLooseMuon && isGlobalMuon'),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    preVtxSelection  = cms.string('abs(userCand("l1").vz - userCand("l2").vz) <= 1.'
                                  '&& 0 < mass() && mass() < 15.0 '
                                  '&& charge() == 0'
                                  '&& userFloat("lep_deltaR") > 0.03'),
    postVtxSelection = cms.string('0 < userFloat("fitted_mass") && userFloat("fitted_mass") < 15.0'
                                  '&& userFloat("sv_prob") > 0.001')
)

CountDiMuonBPH = cms.EDFilter("PATCandViewCountFilter",
    minNumber = cms.uint32(1),
    maxNumber = cms.uint32(999999),
    src = cms.InputTag("MuMu:SelectedDiLeptons")
)  

########################### Tables ###########################

MuMuTable = cms.EDProducer("SimpleCompositeCandidateFlatTableProducer",
    src = cms.InputTag("MuMu:SelectedDiLeptons"),
    cut = cms.string(""), #we should not filter on cross linked collections
    name = cms.string("MuMu"),
    doc  = cms.string("Dilepton collections"),
    singleton = cms.bool(False), # the number of entries is variable
    extension = cms.bool(False), # this is the main table for the muons
    variables = cms.PSet(CandVars,
        l1_idx = Var("userInt('l1_idx')", int, doc = "leading muon index to the BPH muon collection"),
        l2_idx = Var("userInt('l2_idx')", int, doc = "subleading muon index to the BPH muon collection"),
        fit_mass = Var("userFloat('fitted_mass')", float, doc="Fitted dilepton mass"),
        fit_massErr = Var("userFloat('fitted_massErr')", float, doc = "post-fit uncertainty of the mass of the B candidate", precision=12),
        svprob = Var("userFloat('sv_prob')", float, doc="Vtx fit probability"),
        l_xy     = Var("userFloat('l_xy')", float, doc = "post-fit vertex displacement on transverse plane wrt beamspot"),
        l_xy_unc = Var("userFloat('l_xy_unc')", float, doc = "post-fit vertex uncertainty of displacement on transverse plane wrt beamspot"),
        vtx_x =Var("userFloat('vtx_x')", float, doc="Vtx position in x", precision=12),
        vtx_y = Var("userFloat('vtx_y')", float, doc="Vtx position in y", precision=12),
        vtx_z = Var("userFloat('vtx_z')", float, doc="Vtx position in y", precision=12),
        cos2D     = Var("userFloat('cos_theta_2D')", float, doc = "cos 2D of pre-fit candidate wrt beamspot", precision=12),
        fit_cos2D = Var("userFloat('fitted_cos_theta_2D')", float, doc = "cos 2D of fitted vertex wrt beamspot"),

    )
)

MuMuSequence = cms.Sequence(MuMu)
MuMuTables = cms.Sequence(MuMuTable)
