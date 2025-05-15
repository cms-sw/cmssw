import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *

hltPixelTrackTable = cms.EDProducer(
    "SimpleTriggerTrackFlatTableProducer",
    src = cms.InputTag("hltPhase2PixelTracks"),
    cut = cms.string(""),
    name = cms.string("hltPixelTrack"),
    doc = cms.string("HLT Pixel Track information"),
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

hltGeneralTrackTable = cms.EDProducer(
    "SimpleTriggerTrackFlatTableProducer",
    src = cms.InputTag("hltGeneralTracks"),
    cut = cms.string(""),
    name = cms.string("hltGeneralTrack"),
    doc = cms.string("HLT General Track information"),
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
