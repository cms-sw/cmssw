import FWCore.ParameterSet.Config as cms

from DQM.TrackingMonitor.TrackingMonitor_cfi import *
MonitorTrackTKCosmicMuons = TrackMon.clone(
    TrackProducer = 'ctfWithMaterialTracksP5',
    AlgoName = 'ctf',
    FolderName = 'Muons/TKTrack',
    doBeamSpotPlots = False,
    BSFolderName = 'Muons/TKTrack/BeamSpotParameters',
    doSeedParameterHistos = False,
    doAllPlots = False,
    doHitPropertiesPlots = True,
    doGeneralPropertiesPlots = True
)

