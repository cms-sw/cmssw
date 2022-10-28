import FWCore.ParameterSet.Config as cms

from DQM.TrackingMonitor.TrackingMonitor_cfi import *
MonitorTrackGLBCosmicMuons = TrackMon.clone(
    TrackProducer = 'globalCosmicMuons',
    AlgoName = 'glb',
    FolderName = 'Muons/globalCosmicMuons',
    doBeamSpotPlots = False,
    BSFolderName = 'Muons/globalCosmicMuons/BeamSpotParameters',
    doSeedParameterHistos = False,
    doAllPlots = False,
    doHitPropertiesPlots = True,
    doGeneralPropertiesPlots = True,
)
