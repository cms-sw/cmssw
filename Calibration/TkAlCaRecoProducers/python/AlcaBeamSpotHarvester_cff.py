import FWCore.ParameterSet.Config as cms

from Calibration.TkAlCaRecoProducers.AlcaBeamSpotHarvester_cfi import *
alcaBeamSpotHarvesting = cms.Sequence( alcaBeamSpotHarvester )
