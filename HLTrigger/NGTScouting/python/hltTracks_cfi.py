import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *

hltPixelTrackTable = cms.EDProducer(
    "SimpleTriggerTrackFlatTableProducer",
    skipNonExistingSrc = cms.bool(True),
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
        dxyError = Var("dxyError()", "float", doc = "dxyError (cm)"),
        dZError = Var("dzError()", "float", doc = "dzError (cm)"),
        t0 = Var("t0()", "float", doc = "t0 (ns)"),
        vx = Var("vx()", "float", doc = "vx (cm)"),
        vy = Var("vy()", "float", doc = "vy (cm)"),
        vz = Var("vz()", "float", doc = "vz (cm)"),
        charge = Var("charge()", "float", doc = "charge"),
        nPixelHits = Var("hitPattern().numberOfValidPixelHits()", "int16", doc = ""),
        nTrkLays = Var("hitPattern().trackerLayersWithMeasurement()", "int16", doc = ""),
        chi2 = Var("chi2()", "float", doc = "Track chi2"),
        ndof = Var("ndof()", "float", doc = "Number of degrees of freedom"),
        #isLoose = Var("quality('loose')", "bool", doc = "Loose track flag"),
        isTight = Var("quality('tight')", "bool", doc = "Tight track flag"),
        isHighPurity = Var("quality('highPurity')", "bool", doc = "High-purity track flag"),
    )
)

hltPixelTrackExtTable = cms.EDProducer("HLTTracksExtraTableProducer",
                                       tableName = cms.string("hltPixelTrack"),                                    
                                       skipNonExistingSrc = cms.bool(True),
                                       tracksSrc = cms.InputTag("hltPhase2PixelTracks"),
                                       beamSpot = cms.InputTag("hltOnlineBeamSpot"),
                                       precision = cms.int32(7))

hltGeneralTrackTable = cms.EDProducer(
    "SimpleTriggerTrackFlatTableProducer",
    skipNonExistingSrc = cms.bool(True),
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
        vx = Var("vx()", "float", doc = "vx (cm)"),
        vy = Var("vy()", "float", doc = "vy (cm)"),
        vz = Var("vz()", "float", doc = "vz (cm)"),
        charge = Var("charge()", "float", doc = "charge"),
        nPixelHits = Var("hitPattern().numberOfValidPixelHits()", "int16", doc = ""),
        nTrkLays = Var("hitPattern().trackerLayersWithMeasurement()", "int16", doc = ""),
        chi2 = Var("chi2()", "float", doc = "Track chi2"),
        ndof = Var("ndof()", "float", doc = "Number of degrees of freedom"),
    )
)

hltGeneralTrackExtTable = cms.EDProducer("HLTTracksExtraTableProducer",
                                         tableName = cms.string("hltGeneralTrack"),
                                         skipNonExistingSrc = cms.bool(True),
                                         tracksSrc = cms.InputTag("hltGeneralTracks"),
                                         beamSpot = cms.InputTag("hltOnlineBeamSpot"),
                                         precision = cms.int32(7))
