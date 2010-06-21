import FWCore.ParameterSet.Config as cms

from Calibration.TkAlCaRecoProducers.AlcaBeamSpotProducer_cfi import *
alcaBeamSpot = cms.Sequence( alcaBeamSpotProducer )
