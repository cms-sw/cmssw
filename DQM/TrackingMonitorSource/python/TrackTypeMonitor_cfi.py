import FWCore.ParameterSet.Config as cms

from DQM.TrackingMonitorSource.trackTypeMonitor_cfi import trackTypeMonitor
TrackTypeMonitor = trackTypeMonitor.clone(
    trackInputTag      = 'selectedTracks',
    offlineBeamSpot    = 'offlineBeamSpot',
    trackQuality       = 'highPurity',
    vertexTag          = 'selectedPrimaryVertices',
    TrackEtaPar        = dict(Xbins = 60,Xmin = -3.0,Xmax = 3.0),
    TrackPtPar         = dict(Xbins = 100,Xmin = 0.0,Xmax = 100.0),
    TrackPPar          = dict(Xbins = 100,Xmin = 0.0,Xmax = 100.0),
    TrackPhiPar        = dict(Xbins = 100,Xmin = -4.0,Xmax = 4.0),
    TrackPterrPar      = dict(Xbins = 100,Xmin = 0.0,Xmax = 100.0),
    TrackqOverpPar     = dict(Xbins = 100,Xmin = -10.0,Xmax = 10.0),
    TrackdzPar         = dict(Xbins = 100,Xmin = -100.0,Xmax = 100.0),
    TrackChi2bynDOFPar = dict(Xbins = 100,Xmin = 0.0,Xmax = 10.0),
    nTracksPar         = dict(Xbins = 100,Xmin = -0.5,Xmax = 99.5)
)
