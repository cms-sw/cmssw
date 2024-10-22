import FWCore.ParameterSet.Config as cms


muonsPt10 = cms.EDFilter("MuonSelector",
    src = cms.InputTag('muons'),
    cut = cms.string(
        'isGlobalMuon &'
        'isTrackerMuon &'
        'numberOfMatches > 1 &'
        'globalTrack.hitPattern.numberOfValidMuonHits > 0 &'
        'abs(eta) < 2.5 &'
        'pt > 10'
    ),
    filter = cms.bool(False)
)


from SimMuon.MCTruth.MuonTrackProducer_cfi import *
muonInnerTrack = muonTrackProducer.clone(
    #muonsTag = "muons",
    muonsTag = "muonsPt10",
    selectionTags = ('All',),
    trackType = "innerTrack"
)

from DQM.TrackingMonitor.TrackingMonitor_cfi import *
MonitorTrackMuonsInnerTrack = TrackMon.clone(
    TrackProducer = 'muonInnerTrack',
    AlgoName = 'inner',
    FolderName = 'Muons/Tracking/innerTrack',
    doBeamSpotPlots = True,
    BSFolderName = 'Muons/Tracking/innerTrack/BeamSpotParameters',
    doSeedParameterHistos = False,
    doProfilesVsLS = False,
    doAllPlots = False,
    doGeneralPropertiesPlots = True,
    doHitPropertiesPlots = True,
    doTrackerSpecific = True,
    doDCAPlots = True,
    doDCAwrtPVPlots = True,
    doDCAwrt000Plots = False,
    doSIPPlots  = True,
    doEffFromHitPatternVsPU = True,
    doEffFromHitPatternVsBX = False,
    TkSizeBin = 10,
    TkSizeMax = 10.,
    phiErrMax = 0.001,
    etaErrMax = 0.001,
    PVBin = 40,
    PVMin = -0.5,
    PVMax = 79.5, ## it might need to be adjust if CMS asks to have lumi levelling at lower values
    doRecHitVsPhiVsEtaPerTrack = True,
    doRecHitVsPtVsEtaPerTrack = True,
    #doGoodTrackRecHitVsPhiVsEtaPerTrack = True,
    doLayersVsPhiVsEtaPerTrack = True,
    #doGoodTrackLayersVsPhiVsEtaPerTrack = True,
    Eta2DBin = 16,
    Phi2DBin = 16,
    TrackPtBin = 50
)

from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase1Pixel.toModify(MonitorTrackMuonsInnerTrack, EtaBin=31, EtaMin=-3., EtaMax=3.)
phase2_tracker.toModify(MonitorTrackMuonsInnerTrack, EtaBin=46, EtaMin=-4.5, EtaMax=4.5)
phase2_tracker.toModify(MonitorTrackMuonsInnerTrack, PVBin=125, PVMin=-0.5, PVMax=249.5)


#MonitorTrackINNMuons = cms.Sequence(muonInnerTrack+MonitorTrackMuonsInnerTrack)
MonitorTrackINNMuons = cms.Sequence(cms.ignore(muonsPt10)+muonInnerTrack+MonitorTrackMuonsInnerTrack)
