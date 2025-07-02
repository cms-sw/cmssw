import FWCore.ParameterSet.Config as cms
from Validation.RecoMuon.track_selectors_cff import hltPhase2L3MuonIdTracks
from PhysicsTools.NanoAOD.common_cff import *

hltMuonTable = cms.EDProducer(
    "SimpleTriggerTrackFlatTableProducer",
    skipNonExistingSrc = cms.bool(True),
    src = cms.InputTag("hltPhase2L3MuonIdTracks"),
    cut = cms.string(""),
    name = cms.string("hltMuon"),
    doc = cms.string("HLT Muon information"),
    extension = cms.bool(False),
    variables = cms.PSet(
        pt = Var("pt()", "float", doc = "p_T (GeV)"),
        eta = Var("eta()", "float", doc = "#eta"),
        phi = Var("phi()", "float", doc = "#phi (rad)"),
        dXY = Var("dxy()", "float", doc = "dXY (cm)"),
        dZ = Var("dz()", "float", doc = "dZ (cm)"),
        t0 = Var("t0()", "float", doc = "t0 (ns)"),
        nPixelHits = Var("hitPattern().numberOfValidPixelHits()", "int16", doc = ""),
        nTrkLays = Var("hitPattern().trackerLayersWithMeasurement()", "int16", doc = ""),
        nMuHits = Var("hitPattern().numberOfValidMuonHits()", "int16", doc = "")
    )
)
