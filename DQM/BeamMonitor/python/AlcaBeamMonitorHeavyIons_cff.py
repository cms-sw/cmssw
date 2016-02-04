import FWCore.ParameterSet.Config as cms

from DQM.BeamMonitor.AlcaBeamMonitor_cfi import *

AlcaBeamMonitor.PrimaryVertexLabel = 'hiSelectedVertex'
AlcaBeamMonitor.TrackLabel         = 'hiSelectedTracks'
AlcaBeamMonitor.BeamFitter.TrackCollection = 'hiSelectedTracks'
AlcaBeamMonitor.BeamFitter.TrackQuality    = ['highPurity']
AlcaBeamMonitor.PVFitter.VertexCollection  = 'hiSelectedVertex'

alcaBeamMonitor = cms.Sequence( AlcaBeamMonitor )
