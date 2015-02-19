import FWCore.ParameterSet.Config as cms

from DQM.BeamMonitor.AlcaBeamMonitor_cfi import *

AlcaBeamMonitor.PrimaryVertexLabel = 'hiSelectedVertex'
AlcaBeamMonitor.TrackLabel         = 'hiGeneralTracks'
AlcaBeamMonitor.BeamFitter.TrackCollection = 'hiGeneralTracks'
AlcaBeamMonitor.BeamFitter.TrackQuality    = ['highPurity']
AlcaBeamMonitor.PVFitter.VertexCollection  = 'hiSelectedVertex'

import RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi
scalerBeamSpot = RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi.onlineBeamSpotProducer.clone()
alcaBeamMonitor = cms.Sequence( scalerBeamSpot*AlcaBeamMonitor )

