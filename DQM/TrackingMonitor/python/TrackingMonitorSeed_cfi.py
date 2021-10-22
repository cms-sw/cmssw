import FWCore.ParameterSet.Config as cms

from DQM.TrackingMonitor.TrackingMonitor_cfi import *

TrackMonSeed = TrackMon.clone(
    MeasurementState = 'ImpactPoint',
    FolderName = 'Tracking/TrackParameters',
    BSFolderName = 'Tracking/TrackParameters/BeamSpotParameters',
    AlgoName = 'Seed',
    #doGoodTrackPlots = False,
    doTrackerSpecific = False,
    doAllPlots = False,
    doHitPropertiesPlots = False,
    doGeneralPropertiesPlots = False,
    doBeamSpotPlots = False,
    doSeedParameterHistos = False,
    doLumiAnalysis = False,
    doMeasurementStatePlots = False,
    doRecHitsPerTrackProfile = False,
    doRecHitVsPhiVsEtaPerTrack = False,
    #doGoodTrackRecHitVsPhiVsEtaPerTrack = False,
    #
    # plot on Seed (total number, pt, seed # vs cluster)
    #
    doSeedNumberHisto = True,
    doSeedLumiAnalysis = True,
    doSeedVsClusterHisto = True,
    doSeedPTHisto = True,
    doSeedETAHisto = True,
    doSeedPHIHisto = True,
    doSeedPHIVsETAHisto = True,
    doStopSource = True
)
