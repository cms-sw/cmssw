import FWCore.ParameterSet.Config as cms

from DQMOffline.Alignment.tkAlCaRecoMonitor_cfi import tkAlCaRecoMonitor as _tkAlCaRecoMonitor
TkAlCaRecoMonitor = _tkAlCaRecoMonitor.clone(
    TrackProducer = "generalTracks",
    ReferenceTrackProducer = "generalTracks",
    CaloJetCollection = "ak4CaloJets",
    AlgoName = "testTkAlCaReco",
    runsOnReco = False,
    fillInvariantMass = False,
    fillRawIdMap = True,
    useSignedR = False,
    #
    TrackEfficiencyBin = 102,
    TrackEfficiencyMin = -0.01,
    TrackEfficiencyMax = 1.01,
    #
    maxJetPt = 10., #GeV
    #
    SumChargeBin = 11,
    SumChargeMin = -5.5,
    SumChargeMax = 5.5,
    #
    MassBin = 100,
    MassMin = 0.0,
    MassMax = 100.0,
    #
    TrackPtBin = 110,
    TrackPtMin = 0.0,
    TrackPtMax = 110.0,
    #
    TrackCurvatureBin = 2000,
    TrackCurvatureMin = -0.01,
    TrackCurvatureMax = 0.01,
    #
    JetPtBin = 100,
    JetPtMin = 0.0,
    JetPtMax = 50.0,
    #
    MinJetDeltaRBin = 100,
    MinJetDeltaRMin = 0,
    MinJetDeltaRMax = 10,
    #
    MinTrackDeltaRBin = 100,
    MinTrackDeltaRMin = 0,
    MinTrackDeltaRMax = 3.2,
    #
    HitMapsZBin = 300,
    HitMapZMax = 300., #cm
    HitMapsRBin = 120,
    HitMapRMax = 120., #cm
    #
    daughterMass = 0.10565836,#Gev
    FolderName = "TkAlCaRecoMonitor")

