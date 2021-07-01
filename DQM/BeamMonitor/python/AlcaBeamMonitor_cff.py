import FWCore.ParameterSet.Config as cms

from DQM.BeamMonitor.AlcaBeamMonitor_cfi import *
#Check if perLSsaving is enabled to mask MEs vs LS
from DQMServices.Core.DQMStore_cfi import DQMStore
if(DQMStore.saveByLumi):
    AlcaBeamMonitor.perLSsaving=True

import RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi
scalerBeamSpot = RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi.onlineBeamSpotProducer.clone()

alcaBeamMonitor = cms.Sequence( scalerBeamSpot*AlcaBeamMonitor )


