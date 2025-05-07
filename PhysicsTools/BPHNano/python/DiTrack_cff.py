import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *

DiTrack = cms.EDProducer(
    'DiTrackBuilder',
    tracks = cms.InputTag('tracksBPH', 'SelectedTracks'),
    transientTracks = cms.InputTag('tracksBPH', 'SelectedTransientTracks'),
    trk1Selection   = cms.string(''),
    trk2Selection   = cms.string(''),
    trk1Mass = cms.double(0.139),
    trk2Mass = cms.double(0.494),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    preVtxSelection = cms.string('((userFloat("unfitted_mass_KK")>0.95 && userFloat("unfitted_mass_KK")<1.12) || (userFloat("unfitted_mass_Kpi")>0.6 && userFloat("unfitted_mass_Kpi")<1.2) || (userFloat("unfitted_mass_piK")>0.6 && userFloat("unfitted_mass_piK")<1.2)) && charge() == 0'),
    postVtxSelection =  cms.string('((userFloat("fitted_mass_KK")>0.95 && userFloat("fitted_mass_KK")<1.12) || (userFloat("fitted_mass_Kpi")>0.6 && userFloat("fitted_mass_Kpi")<1.2)  || (userFloat("fitted_mass_piK")>0.6 && userFloat("fitted_mass_piK")<1.2)) && userFloat("sv_prob") > 0.001')
)

CountDiTrack = cms.EDFilter("PATCandViewCountFilter",
    minNumber = cms.uint32(1),
    maxNumber = cms.uint32(999999),
    src       = cms.InputTag("DiTrack")
)  

DiTrackTable = cms.EDProducer("SimpleCompositeCandidateFlatTableProducer",
    src  = cms.InputTag("DiTrack"),
    cut  = cms.string(""), #we should not filter on cross linked collections
    name = cms.string("DiTrack"),
    doc  = cms.string("slimmedDiTrack for BPark after basic selection"),
    singleton = cms.bool(False), # the number of entries is variable
    extension = cms.bool(False), # this is the main table for the muons
    variables = cms.PSet(
        CandVars,
        fit_mass_KK = Var("userFloat('fitted_mass_KK')", float, doc="", precision=6),
        fit_mass_Kpi = Var("userFloat('fitted_mass_Kpi')", float, doc="", precision=6),
        fit_mass_piK = Var("userFloat('fitted_mass_piK')", float, doc="", precision=6),        
        fit_pt   = Var("userFloat('fitted_pt')", float, doc="", precision=6),
        fit_eta  = Var("userFloat('fitted_eta')", float, doc="", precision=6),
        fit_phi  = Var("userFloat('fitted_phi')", float, doc="", precision=6),
        svprob      = Var("userFloat('sv_prob')", float, doc="", precision=6),
        trk1_idx    = Var("userInt('trk1_idx')", int, doc=""),
        trk2_idx    = Var("userInt('trk2_idx')", int, doc=""),
        vtx_x       = Var("userFloat('vtx_x')", float, doc="", precision=6),
        vtx_y       = Var("userFloat('vtx_y')", float, doc="", precision=6),
        vtx_z       = Var("userFloat('vtx_z')", float, doc="", precision=6),     
        l_xy        = Var("userFloat('l_xy')", float, doc="", precision=6),
        l_xy_unc        = Var("userFloat('l_xy_unc')", float, doc="", precision=6),
        cos_theta_2D    = Var("userFloat('fitted_cos_theta_2D')", float, doc="", precision=6),
        sv_prob         = Var("userFloat('sv_prob')", float, doc="", precision=6),
        sv_ndof         = Var("userFloat('sv_ndof')", float, doc="", precision=6),
        sv_chi2         = Var("userFloat('sv_chi2')", float, doc="", precision=6),
        vtx_cxx = Var("userFloat('vtx_cxx')", float, doc="", precision=6),
        vtx_cyy = Var("userFloat('vtx_cyy')", float, doc="", precision=6),
        vtx_czz = Var("userFloat('vtx_czz')", float, doc="", precision=6),
        vtx_cyx = Var("userFloat('vtx_cyx')", float, doc="", precision=6),
        vtx_czx = Var("userFloat('vtx_czx')", float, doc="", precision=6),
        vtx_czy = Var("userFloat('vtx_czy')", float, doc="", precision=6)

    )
)

DiTrackSequence = cms.Sequence(DiTrack)
DiTrackTables   = cms.Sequence(DiTrackTable)
