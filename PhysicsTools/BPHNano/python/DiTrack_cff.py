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

DiTrackSequence = cms.Sequence(DiTrack)
