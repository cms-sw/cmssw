import FWCore.ParameterSet.Config as cms

from DQM.BeamMonitor.AlcaBeamMonitor_cfi import *
#Check if perLSsaving is enabled to mask MEs vs LS
from Configuration.ProcessModifiers.dqmPerLSsaving_cff import dqmPerLSsaving
dqmPerLSsaving.toModify(AlcaBeamMonitor, perLSsaving=True)

from RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi import onlineBeamSpotProducer as _onlineBeamSpotProducer
scalerBeamSpot = _onlineBeamSpotProducer.clone()

alcaBeamMonitor = cms.Sequence( scalerBeamSpot*AlcaBeamMonitor )


