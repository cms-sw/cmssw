import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *

from PhysicsTools.NanoAOD.trackingAssocValueMapsProducer_cfi import trackingAssocValueMapsProducer

tpSelectorPixelTracks = cms.PSet(
    lip = cms.double(30.0),
    chargedOnly = cms.bool(True),
    stableOnly = cms.bool(False),
    pdgId = cms.vint32(),
    signalOnly = cms.bool(False),
    intimeOnly = cms.bool(True),
    minRapidity = cms.double(-4.5),
    minHit = cms.int32(0),
    ptMin = cms.double(0.9),
    ptMax = cms.double(1e100),
    maxRapidity = cms.double(4.5),
    tip = cms.double(2.5),
    invertRapidityCut = cms.bool(False),
    minPhi = cms.double(-3.2),
    maxPhi = cms.double(3.2),
)

pixelTrackAssoc = trackingAssocValueMapsProducer.clone(
    trackCollection =  cms.InputTag("hltPhase2PixelTracks"),
    associator = cms.InputTag("hltTrackAssociatorByHits"),
    trackingParticles = cms.InputTag("mix", "MergedTrackTruth"),
    tpSelectorPSet = tpSelectorPixelTracks,
    storeTPKinematics = cms.bool(True),
    useMuonAssociators = cms.bool(False)
)

from Configuration.ProcessModifiers.phase2CAExtension_cff import phase2CAExtension
phase2CAExtension.toModify(pixelTrackAssoc, trackCollection = "hltPhase2PixelTracksCAExtension")

hltPixelTrackTable = cms.EDProducer(
    "SimpleTriggerTrackFlatTableProducer",
    skipNonExistingSrc = cms.bool(True),
    src = cms.InputTag("hltPhase2PixelTracks"),
    cut = cms.string(""),
    name = cms.string("hltPixelTrack"),
    doc = cms.string("HLT Pixel Track information"),
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
        qoverp = Var("qoverp()", "float", doc = "q/p"),
        dsz = Var("dsz()", "float", doc = "dsz (cm)"),
        qoverpErr = Var("qoverpError()", "float", doc = ""),
        ptErr = Var("ptError()", "float", doc = ""),
        lambdaErr = Var("lambdaError()", "float", doc = ""),
        dszErr = Var("dszError()", "float", doc = ""),
        etaErr = Var("etaError()", "float", doc = ""),
        phiErr = Var("phiError()", "float", doc = ""),
    ),
        externalVariables = cms.PSet(
        matched   = cms.PSet(src = cms.InputTag("pixelTrackAssoc","matched"),
                             doc = cms.string("1 if matched to a TrackingParticle"),
                             type = cms.string("uint8")),
        duplicate = cms.PSet(src = cms.InputTag("pixelTrackAssoc","duplicate"),
                             doc = cms.string("1 if multiple reco tracks map to same TP"),
                             type = cms.string("uint8")),
        tpPdgId  = cms.PSet(src = cms.InputTag("pixelTrackAssoc","tpPdgId"),
                             doc = cms.string("pdgId of matched TrackingParticle"),
                             type = cms.string("int16")),
        tpCharge = cms.PSet(src = cms.InputTag("pixelTrackAssoc","tpCharge"),
                             doc = cms.string("charge of matched TrackingParticle"),
                             type = cms.string("int16")),
        tpPt     = cms.PSet(src = cms.InputTag("pixelTrackAssoc","tpPt"),
                             doc = cms.string("pt of matched TrackingParticle"),
                             type = cms.string("float")),
        tpEta    = cms.PSet(src = cms.InputTag("pixelTrackAssoc","tpEta"),
                             doc = cms.string("eta of matched TrackingParticle"),
                             type = cms.string("float")),
        tpPhi    = cms.PSet(src = cms.InputTag("pixelTrackAssoc","tpPhi"),
                             doc = cms.string("phi of matched TrackingParticle"),
                             type = cms.string("float"))
    )
)

hltPixelTrackExtTable = cms.EDProducer("HLTTracksExtraTableProducer",
                                       tableName = cms.string("hltPixelTrack"),                                    
                                       skipNonExistingSrc = cms.bool(True),
                                       tracksSrc = cms.InputTag("hltPhase2PixelTracks"),
                                       beamSpot = cms.InputTag("hltOnlineBeamSpot"),
                                       precision = cms.int32(7))

hltPixelTrackRecHitsTable = cms.EDProducer("HLTTracksRecHitsTableProducer",
                                            tableName = cms.string("hltPixelTrackRecHits"),
                                            skipNonExistingSrc = cms.bool(True),
                                            tracksSrc = cms.InputTag("hltPhase2PixelTracks"),
                                            maxRecHits = cms.uint32(16),
                                            precision = cms.int32(7)
)


phase2CAExtension.toModify(hltPixelTrackTable, src = "hltPhase2PixelTracksCAExtension")
phase2CAExtension.toModify(hltPixelTrackExtTable, tracksSrc = "hltPhase2PixelTracksCAExtension")
phase2CAExtension.toModify(hltPixelTrackRecHitsTable, tracksSrc = "hltPhase2PixelTracksCAExtension")

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
