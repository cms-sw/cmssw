import FWCore.ParameterSet.Config as cms

from DQM.TrackingMonitor.TrackingMonitor_cfi import *
MonitorTrackGLBMuons = TrackMon.clone(
    TrackProducer = 'globalMuons',
    AlgoName = 'glb',
    FolderName = 'Muons/globalMuons',
    doBeamSpotPlots = False,
    BSFolderName = 'Muons/globalCosmicMuons/BeamSpotParameters',
    doSeedParameterHistos = False,
    doProfilesVsLS = True,
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
    doEffFromHitPatternVsLUMI = True
)
