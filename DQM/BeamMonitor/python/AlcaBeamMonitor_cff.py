import FWCore.ParameterSet.Config as cms

from DQM.BeamMonitor.AlcaBeamMonitor_cfi import *
#Check if perLSsaving is enabled to mask MEs vs LS
from Configuration.ProcessModifiers.perLSsaving_cff import perLSsaving
perLSsaving.toModify(AlcaBeamMonitor, perLSsaving=True)

import RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi
scalerBeamSpot = RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi.onlineBeamSpotProducer.clone()

alcaBeamMonitor = cms.Sequence( scalerBeamSpot*AlcaBeamMonitor )


