import FWCore.ParameterSet.Config as cms

from DQM.BeamMonitor.AlcaBeamMonitor_cfi import *

AlcaBeamMonitor.PrimaryVertexLabel = 'hiSelectedVertex'
AlcaBeamMonitor.TrackLabel         = 'hiGeneralTracks'
AlcaBeamMonitor.BeamFitter.TrackCollection = 'hiGeneralTracks'
AlcaBeamMonitor.BeamFitter.TrackQuality    = ['highPurity']
AlcaBeamMonitor.PVFitter.VertexCollection  = 'hiSelectedVertex'
#Check if perLSsaving is enabled to mask MEs vs LS
from Configuration.ProcessModifiers.dqmPerLSsaving_cff import dqmPerLSsaving
dqmPerLSsaving.toModify(AlcaBeamMonitor, perLSsaving=True)

import RecoVertex.BeamSpotProducer.onlineBeamSpotESProducer_cfi as _mod
BeamSpotESProducer = _mod.onlineBeamSpotESProducer.clone(
    timeThreshold = 999999 # accept even old payloads for MC/Data ReReco. DO NOT CHANGE
)

import RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi
scalerBeamSpot = RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi.onlineBeamSpotProducer.clone()
alcaBeamMonitor = cms.Sequence( scalerBeamSpot*AlcaBeamMonitor )

